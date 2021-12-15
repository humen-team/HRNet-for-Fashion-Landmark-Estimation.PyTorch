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
import sys
sys.path.append('/home/ubuntu/py-thin-plate-spline')
import thinplate as tps
# from kornia.geometry.transform import get_tps_transform, warp_image_tps

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

UV_LANDMARKS_PATH = 'tshirt_uv_landmarks.pth'
UV_SEGMASK_PATHS = dict(
    tshirt = dict(
        core_front='tshirt_uv_segs/tshirt_uv_seg_front.png',
        sleeve_left_front='tshirt_uv_segs/tshirt_uv_seg_front_left.png',
        sleeve_right_front='tshirt_uv_segs/tshirt_uv_seg_front_right.png',
        core_back='tshirt_uv_segs/tshirt_uv_seg_back.png',
        sleeve_left_back='tshirt_uv_segs/tshirt_uv_seg_back_left.png',
        sleeve_right_back='tshirt_uv_segs/tshirt_uv_seg_back_right.png',
    )
)
TEXTURE_RESOLUTION = (512, 512)

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    
    parser.add_argument('--front_input',
                        '--front-input',
                        help='path to the front input file',
                        type=str,
                        required=True)
    parser.add_argument('--back_input',
                        '--back-input',
                        help='path to the back input file',
                        type=str,
                        required=True)
    parser.add_argument('--output_dir',
                        help='path to the output file',
                        type=str,
                        default='results/')
    parser.add_argument("--use_heuristics",
                        "--use-heuristics",
                        action="store_true",
                        help="if specified, use heuristics to post process predictions",)

    # ****** can ignore the rest of the args for hack week ******
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


    args = parser.parse_args()
    return args


def insensitive_glob(pattern):
    def either(c):
        return '[%s%s]' % (c.lower(), c.upper()) if c.isalpha() else c
    return glob(''.join(map(either, pattern)))

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


def enforce_heuristics(joints, joints_vis, viewpoint):
    flip_pairs = [[2,6],[3,5],[7,25],[8,24],[9,23],[10,22],[11,21],[12,20],[13,19],[14,18],[15,17]]
    for fp in flip_pairs:
        if viewpoint == 'front':
            left_joint = fp[0] - 1
            right_joint = fp[1] - 1
        elif viewpoint == 'back':
            left_joint = fp[1] - 1
            right_joint = fp[0] - 1

        if joints[left_joint, 0] > joints[right_joint, 0]:
            # flip when the left joint is detected on the right
            tmp = np.copy(joints[left_joint])
            joints[left_joint] = joints[right_joint]
            joints[right_joint] = tmp

            tmp = np.copy(joints_vis[left_joint])
            joints_vis[left_joint] = joints_vis[right_joint]
            joints_vis[right_joint] = tmp

    return joints, joints_vis

def save_prediction(input_image, joints, joints_vis, output_dir, output_prefix, use_heuristics=False, vis_thresh=0.5):
    os.makedirs(output_dir, exist_ok=True)

    if use_heuristics:
        if 'front' in output_prefix or 'Front' in output_prefix:
            joints, joints_vis = enforce_heuristics(joints, joints_vis, 'front')
        elif 'back' in output_prefix or 'Back' in output_prefix:
            joints, joints_vis = enforce_heuristics(joints, joints_vis, 'back')
        else:
            print("To use heuristics, 'front/Front' or 'back/Back' needs to be in file name")
            raise

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

def load_uv_data(texture_resolution):
    tshirt_uv_landmarks = torch.load(UV_LANDMARKS_PATH)
    tshirt_uv_segmasks = {
        region: cv2.resize(cv2.imread(segmask_path), texture_resolution)
        for region, segmask_path in UV_SEGMASK_PATHS['tshirt'].items()
    }
    return dict(
        landmarks=tshirt_uv_landmarks,
        segments=tshirt_uv_segmasks,
    )


def warp_to_uv(photos, landmarks, texture_resolution=(512,512)):

    # TODO: This mapping is specific to tshirts; In the future we need a more general solution
    region_to_side = dict(
        core_front='front', sleeve_left_front='front', sleeve_right_front='front',
        core_back='back', sleeve_left_back='back', sleeve_right_back='back',
    )

    # Allowing too many indices to define TPS transformation can make warp finicky
    # TODO:  Remove the need for hardcoing these indices via heuristics
    valid_indices = dict(
        core_front=[2, 4, 6, 7, 14, 15, 17, 18, 25],
        sleeve_left_front=[20, 22, 23, 25],
        sleeve_right_front=[7, 9, 10, 12],
        # core_back=[1, 2, 6, 7, 14, 15, 17, 18, 25],
        core_back=[1, 7, 15, 17, 18, 25],
        sleeve_left_back=[20, 22, 23, 25],
        sleeve_right_back=[7, 9 , 10, 12],
    )

    uv_data = load_uv_data(texture_resolution)

    uv_canvas = torch.zeros(TEXTURE_RESOLUTION[1], TEXTURE_RESOLUTION[0], 3).byte()
    for clothing_region in uv_data['landmarks']:
        photo_side = region_to_side[clothing_region]

        # Only include valid landmarks that are available in both the predictions from photo and the destination uv-map
        valid_photo_landmarks = {k:v[:2] for k,v in landmarks[photo_side].items() if k in valid_indices[clothing_region]}
        valid_uv_landmarks = {k:v for k,v in uv_data['landmarks'][clothing_region].items() if k in valid_indices[clothing_region]}

        # Prepare valid landmarks for warp
        anchors_src = torch.stack(list(valid_photo_landmarks.values()), dim=0)
        anchors_dst = torch.stack(list(valid_uv_landmarks.values()), dim=0)
        
        photo_numpy = (photos[photo_side]/255.0).squeeze(0).permute(1,2,0).numpy()
        warped_image = warp_image_tps(photo_numpy, anchors_src.numpy(), anchors_dst.numpy(), dshape=texture_resolution)
        warped_image = (np.clip(warped_image, 0, 1) * 255).astype(np.uint8)

        # Mask out 
        uv_canvas = uv_canvas * (uv_data['segments'][clothing_region] == 0) + warped_image * (uv_data['segments'][clothing_region] > 0)

    return uv_canvas

def warp_image_tps(img, c_src, c_dst, dshape=None):
    dshape = dshape or img.shape
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps.tps_grid(theta, c_dst, dshape)
    mapx, mapy = tps.tps_grid_to_remap(grid, img.shape)
    return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)

def convert_outputs_to_uv(
    front_image_file, back_image_file, front_landmarks_file, back_landmarks_file, output_file):
    input_image_paths = dict(
        tshirt=dict(
            front=front_image_file,
            back=back_image_file,
        )
    )
    input_landmark_paths = dict(
        tshirt=dict(
            front=front_landmarks_file,
            back=back_landmarks_file,
        )
    )
    
    tshirt_photos = {
        side: torch.from_numpy(cv2.resize(cv2.imread(image_path), TEXTURE_RESOLUTION).transpose(2,0,1)[[2,1,0]]).unsqueeze(0)
        for side, image_path in input_image_paths['tshirt'].items()
    }
    
    tshirt_photos_landmarks = {}
    for side, landmarks_path in input_landmark_paths['tshirt'].items():
        landmarks = torch.from_numpy(np.load(landmarks_path))
        tshirt_photos_landmarks[side] = {}
        for index, landmark in enumerate(landmarks):
            tshirt_photos_landmarks[side][index+1] = landmark
    uv_texture = warp_to_uv(tshirt_photos, tshirt_photos_landmarks, TEXTURE_RESOLUTION)
    cv2.imwrite(output_file, uv_texture[:,:,[2,1,0]].numpy())

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
    
    imfiles = [args.front_input, args.back_input]
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
                args.use_heuristics,
            )

    front_landmarks_file = f"{args.output_dir}/{os.path.basename(args.front_input).split('.')[0]}_landmarks.npy"
    back_landmarks_file = f"{args.output_dir}/{os.path.basename(args.back_input).split('.')[0]}_landmarks.npy"
    uv_output_file = f"{args.output_dir}/uv_output.png"
    convert_outputs_to_uv(
        args.front_input,
        args.back_input,
        front_landmarks_file,
        back_landmarks_file,
        uv_output_file,
    )

if __name__ == '__main__':
    main()
