import argparse
import os

from glob import glob
from subprocess import run

parser = argparse.ArgumentParser(description='Run demo on a folder of examples')

parser.add_argument('--input_dir',
                    help='path to the input dir',
                    type=str,
                    default='samples/')

parser.add_argument('--output_dir',
                    help='path to the output dir',
                    type=str,
                    default='results/')

parser.add_argument("--use_heuristics",
                    "--use-heuristics",
                    action="store_true",
                    help="if specified, use heuristics to post process predictions",)

def insensitive_glob(pattern):
    def either(c):
        return '[%s%s]' % (c.lower(), c.upper()) if c.isalpha() else c
    return glob(''.join(map(either, pattern)))

def parse_back_file(front_file):
    back_file = front_file.replace("front", "back")
    if not os.path.isfile(back_file):
        back_file = front_file.replace("Front", "Back")
    return back_file

def main():
    args = parser.parse_args()
    front_files = insensitive_glob(f"{args.input_dir}/*front*")
    back_files = [parse_back_file(ff) for ff in front_files]
    for front_file, back_file in zip(front_files, back_files):
        input_base_name = os.path.basename(front_file).split('.')[0].replace('front', '').replace('Front', '')
        curr_output_dir = f"{args.output_dir}/{input_base_name}"
        cmd = f"python3 tools/demo.py --front_input {front_file} --back_input {back_file} --output_dir {curr_output_dir} "
        if args.use_heuristics:
            cmd += " --use_heuristics "
        run(cmd, shell=True)

if __name__ == '__main__':
    main()
