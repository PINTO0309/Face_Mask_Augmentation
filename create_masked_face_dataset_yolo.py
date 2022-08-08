import os
import cv2
import pandas as pd
import numpy as np
import glob
import random
random.seed(0)
import shutil
from tqdm import tqdm
from natsort import natsorted
from FaceMasking import FaceMasker
from ibug.face_detection import RetinaFacePredictor
from ibug.face_alignment import FANPredictor

face_detector = RetinaFacePredictor(
    threshold=0.8,
    device='cuda:0',
    model=RetinaFacePredictor.get_model('resnet50')
)
landmark_detector = FANPredictor(
    device='cuda:0',
    model=FANPredictor.get_model('2dfan4')
)


source_folder = 'data/300wlp-640x480'
output_folder = 'data/300wlp-640x480_masked'

if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)

masker = FaceMasker()

less_filfol = []
morethanone = []
nondetected = []

image_files = natsorted(glob.glob(f"{source_folder}/*.jpg"))
text_files = natsorted(glob.glob(f"{source_folder}/*.txt"))

train_txt = ''
test_txt = ''

if f'{source_folder}/train.txt' in text_files:
    text_files.remove(f'{source_folder}/train.txt')
    with open(glob.glob(f"{source_folder}/train.txt")[0], 'r') as f:
        train_txt = [l.strip() for l in f.readlines()]

if f'{source_folder}/test.txt' in text_files:
    text_files.remove(f'{source_folder}/test.txt')
    with open(glob.glob(f"{source_folder}/test.txt")[0], 'r') as f:
        test_txt = [l.strip() for l in f.readlines()]

if f'{source_folder}/val.txt' in text_files:
    text_files.remove(f'{source_folder}/val.txt')

assert len(image_files) == len(text_files), \
    f"len(image_files) != len(text_files): {len(image_files)} {len(text_files)}"


output_train_txt_list = []
output_test_txt_list = []

for j, (image_file, text_file) in tqdm(enumerate(zip(image_files, text_files))):
    image_basename = os.path.basename(image_file)
    image_basename_without_ext = os.path.splitext(image_basename)[0]
    text_basename = os.path.basename(text_file)
    text_basename_without_ext = os.path.splitext(text_basename)[0]
    assert image_basename_without_ext == text_basename_without_ext, \
        f"image_basename_without_ext != text_basename_without_ext: \
            {image_basename_without_ext} {text_basename_without_ext}"

    # Load image
    image = cv2.imread(image_file)[..., ::-1]
    width = image.shape[1]
    height = image.shape[0]

    detected_faces = face_detector(image, rgb=True)

    if len(detected_faces) == 0:
        nondetected.append(image_file)
        continue
    if len(detected_faces) > 1:
        morethanone.append(image_file)
        continue

    landmarks, scores = landmark_detector(image, detected_faces, rgb=False)
    landmarks = [tuple(landmark.astype(np.int32)) for landmark in landmarks[0]]

    # Save masked-extracted face.
    image_mask = masker.wear_mask_to_face(image, landmarks)
    face_mask = image_mask[:, :, ::-1]

    cv2.imwrite(
        f'{output_folder}/{image_basename_without_ext}_masked.jpg',
        face_mask,
    )
    shutil.copy(
        text_file,
        f'{output_folder}/{text_basename_without_ext}_masked.txt',
    )

    if image_file in test_txt:
        output_test_txt_list.append(f'{output_folder}/{image_basename_without_ext}_masked.jpg')
    else:
        output_train_txt_list.append(f'{output_folder}/{image_basename_without_ext}_masked.jpg')


set_path = f'{output_folder}/test.txt'
with open(set_path, 'w') as fset:
    for jpg in output_test_txt_list:
        fset.write(f'{jpg}\n')

set_path = f'{output_folder}/train.txt'
with open(set_path, 'w') as fset:
    for jpg in output_train_txt_list:
        fset.write(f'{jpg}\n')

nondetected = pd.DataFrame(nondetected)
nondetected.to_csv(f'non_detected.csv', header=None, index=None)
morethanone = pd.DataFrame(morethanone)
morethanone.to_csv(f'multi_detected.csv', header=None, index=None)