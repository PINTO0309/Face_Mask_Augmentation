import os
import argparse
import numpy as np
import scipy.io as sio
from tqdm import tqdm

def get_ypr_from_mat(mat_path):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_dir',
        help='root directory of the datasets files',
        default='./datasets/300W_LP',
        type=str
    )
    parser.add_argument(
        '--file_name',
        help='Output filename.',
        default='files.txt',
        type=str
    )
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    os.chdir(args.root_dir)

    file_counter = 0
    rej_counter = 0
    outfile = open(args.file_name, 'w')

    for root, dirs, files in tqdm(os.walk('.'), dynamic_ncols=True):
        for f in tqdm(files, dynamic_ncols=True):
            if f[-4:] == '.jpg':
                mat_path = os.path.join(root, f.replace('.jpg', '.mat'))
                # We get the pose in radians
                pose = get_ypr_from_mat(mat_path)
                # And convert to degrees.
                pitch = pose[0] * 180 / np.pi
                yaw = pose[1] * 180 / np.pi
                roll = pose[2] * 180 / np.pi

                if abs(pitch) <= 99 and abs(yaw) <= 99 and abs(roll) <= 99:
                    if file_counter > 0:
                        outfile.write('\n')
                    outfile.write(root + '/' + f[:-4])
                    file_counter += 1
                else:
                    rej_counter += 1

    outfile.close()
    print(f'{file_counter} files listed! {rej_counter} files had out-of-range values and kept out of the list!')
