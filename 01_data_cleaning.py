import glob
import os
import shutil
from tqdm import tqdm
from natsort import natsorted

OUTPUT_PATH = 'train_aug_120x120_part_masked_clean'
os.makedirs(OUTPUT_PATH, exist_ok=True)

file_list1 = natsorted(glob.glob('train_aug_120x120_part/*/*.jpg'))
file_list2 = natsorted(glob.glob('train_aug_120x120_part_masked/*/*.jpg'))

with open('3dmm_data/new_train_aug_120x120.list.train') as f:
    train_list = f.read().splitlines()
print(f'train_list: {len(train_list)}')

clean_list_count = 0
new_train_list = []
for file_path1, file_path2 in tqdm(zip(file_list1, file_list2)):
    if os.path.basename(file_path1) in train_list:
        shutil.copy2(file_path1, OUTPUT_PATH)
        new_train_list.append(os.path.basename(file_path1))
        clean_list_count += 1

    if '_'.join(os.path.splitext(os.path.basename(file_path2))[0].split('_')[:-2]) + '.jpg' in train_list:
        shutil.copy2(file_path2, OUTPUT_PATH)
        new_train_list.append(os.path.basename(file_path2))
        clean_list_count += 1

# clean_list: 1272504 -> 636252 x2
print(f'clean_list: {clean_list_count}')

new_train_list = '\n'.join(new_train_list)
with open('3dmm_data/new_new_train_aug_120x120.list.train', 'w') as f:
    f.write(new_train_list)

