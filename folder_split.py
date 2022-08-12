import os
import glob
import shutil
from tqdm import tqdm
from natsort import natsorted
from argparse import ArgumentParser

FOLDER_MAX = 2500

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-i',
        '--image_folder_path',
        type=str,
        default='HELEN',
    )
    args = parser.parse_args()

    image_files = natsorted(glob.glob(f"{args.image_folder_path}/*.jpg"))
    mat_files = natsorted(glob.glob(f"{args.image_folder_path}/*.mat"))

    assert len(image_files) == len(mat_files)

    image_count = 0
    for (image_file, mat_file) in tqdm(zip(image_files, mat_files), dynamic_ncols=True):
        new_folder_number = image_count // FOLDER_MAX
        dirname = os.path.dirname(image_file)
        # print(f'@@@ dirname: {dirname} split: {dirname.split("/")}')
        new_dirname = f'{args.image_folder_path}_{str(new_folder_number).zfill(6)}'
        os.makedirs(new_dirname, exist_ok=True)

        shutil.move(image_file, new_dirname)
        shutil.move(mat_file, new_dirname)

        image_count += 1


    print(f'image_count: {image_count}')
    print(f'folder_count: {image_count//FOLDER_MAX+1}')

if __name__ == "__main__":
    main()