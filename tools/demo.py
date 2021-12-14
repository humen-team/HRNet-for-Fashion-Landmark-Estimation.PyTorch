# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cv2
import numpy as np
import os
import pprint
from glob import glob

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from core.inference import get_final_preds
from utils.utils import create_logger

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='experiments/deepfashion2/hrnet/w48_384x288_adam_lr1e-3.yaml',
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    parser.add_argument('--input_dir',
                        help='path to the input file',
                        type=str,
                        default='samples/')
    parser.add_argument('--output_dir',
                        help='path to the output file',
                        type=str,
                        default='results/')

    args = parser.parse_args()
    return args


def load_image(input_file, width, height):
    img = cv2.imread(input_file)  # reads an image in the BGR format
    img = cv2.resize(img, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    img = img.astype(float) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean[None, None, :]) / std[None, None, :]
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).cuda().float()
    return img_tensor


def save_prediction(input_image, joints, joints_vis, output_dir, output_prefix, vis_thresh=0.5):
    os.makedirs(output_dir, exist_ok=True)
    # save the numerical predictions
    width = input_image.shape[2]
    height = input_image.shape[1]
    out = np.concatenate((joints, joints_vis), axis=1)[:25]
    out[:, 0] = out[:, 0] / width
    out[:, 1] = out[:, 1] / height
    np.save(f'{output_dir}/{output_prefix}_landmarks.npy', out)

    # save the visualization

    # mask out non-tshirt landmarks
    # category/landmark index reference:
    # https://github.com/switchablenorms/DeepFashion2/blob/master/evaluation/deepfashion2_to_coco.py
    joints_vis[25:] = 0
    joints_vis[:25] = 1

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_numpy = input_image.cpu().numpy() * std[:, None, None] + mean[:, None, None]
    image_numpy = np.transpose(
        np.clip(image_numpy * 255.0, 0, 255).astype(np.uint8), (1, 2, 0)
    )
    
    image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
    index = 0
    for joint, joint_vis in zip(joints, joints_vis):
        index += 1
        joint_0 = joint[0]
        joint_1 = joint[1]
        if joint_vis[0] > vis_thresh:
            cv2.circle(image_numpy, (int(joint_0), int(joint_1)), 2, [255, 0, 0], 2)
            cv2.putText(image_numpy, f"{index}", (int(joint_0), int(joint_1)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    output_file = f'{output_dir}/{output_prefix}_output.jpg'
    cv2.imwrite(output_file, image_numpy)

def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        batch_size = cfg.TEST.BATCH_SIZE_PER_GPU * torch.cuda.device_count()
        logger.info("Let's use %d GPUs!" % torch.cuda.device_count())

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)  # False
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    # model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model = torch.nn.DataParallel(model).cuda()
    
    input_extensions = ['jpg', 'png']
    imfiles = []
    for ext in input_extensions:
        imfiles += sorted(glob(f"{args.input_dir}/*.{ext}"))
            
    for index, input_file in enumerate(imfiles):
        print(f"Processing: {input_file} ({index}/{len(imfiles)})")
        input = load_image(input_file, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
        with torch.no_grad():
            output = model(input)
            # TODO: if needed, there's a flip prediction and output averaging 
            # implemented in the original code as well

            # block irrelevant channels in output
            preds, maxvals = get_final_preds(cfg, output.detach().cpu().numpy())

            coeff = cfg.MODEL.IMAGE_SIZE[0] / cfg.MODEL.HEATMAP_SIZE[0]
            output_prefix = os.path.basename(input_file).split('.')[0]
            save_prediction(
                input[0], preds[0]*coeff, maxvals[0], args.output_dir, output_prefix,
            )

if __name__ == '__main__':
    main()
