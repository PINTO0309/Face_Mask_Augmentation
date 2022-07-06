import os
import cv2
import time
import pandas as pd
import numpy as np
import glob
import scipy.io as sio
import random
random.seed(0)
import shutil
from FaceMasking import FaceMasker
from natsort import natsorted
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


source_folder = '300W_LP_croped'
output_folder = '300W_LP_croped_masked'
image_size = 480

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

masker = [FaceMasker()]

list_fol = os.listdir(source_folder)
num_fol = len(list_fol)
less_filfol = []
morethanone = []
nondetected = []

st = time.time()
for i, fol in enumerate(list_fol):
    print('folder: {}/{}'.format(i+1, num_fol))
    folfil = os.listdir(os.path.join(source_folder, fol))
    num_fil = len(folfil)
    if len(folfil) <= 0:
        less_filfol.append(fol)
        continue

    image_files = glob.glob(f"{os.path.join(source_folder, fol)}/*.jpg")
    mat_files = glob.glob(f"{os.path.join(source_folder, fol)}/*.mat")

    for j, (image_file, mat_file) in enumerate(zip(natsorted(image_files), natsorted(mat_files))):

        save_fol = f'{output_folder}/{fol}'
        if not os.path.exists(save_fol):
            os.makedirs(save_fol)

        print('  file: {}/{}'.format(j+1, num_fil))

        # Load image
        image = cv2.imread(image_file)[:, :, ::-1]
        # Load .mat
        mat = sio.loadmat(mat_file)
        # 最終クロップ用画像サイズの計算（アノテーションデータのサイズに調整するためのクロップ用）
        pt2d = mat['pt2d']
        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])
        k = 0.20
        x_min -= 2 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 2 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        x_min = max(int(x_min), 0)
        y_min = max(int(y_min), 0)
        x_max = min(int(x_max), int(image_size))
        y_max = min(int(x_max), int(image_size))
        crop_start_x = 0
        crop_end_x = int(x_max-x_min)
        crop_start_y = 0
        crop_end_y = int(y_max-y_min)
        width = image.shape[1]
        height = image.shape[0]

        detected_faces = face_detector(image, rgb=True)
        if len(detected_faces) == 0:
            detected_faces = np.asarray([[0,0,image.shape[1],image.shape[0],1.0]])
        landmarks, scores = landmark_detector(image, detected_faces, rgb=False)
        landmarks = [tuple(landmark.astype(np.int32)) for landmark in landmarks[0]]

        if len(detected_faces) == 0:
            nondetected.append(image_file)
            continue
        if len(detected_faces) > 1:
            morethanone.append(image_file)
            continue

        # Save masked-extracted face.
        mkr = masker[0]
        image_mask = mkr.wear_mask_to_face(image, landmarks)
        face_mask = image_mask[:, :, ::-1]

        cv2.imwrite(
            os.path.join(save_fol, os.path.basename(image_file).split('.')[0] + f'.jpg'),
            image[crop_start_y:crop_end_y, crop_start_x:crop_end_x, :][..., ::-1],
        )
        cv2.imwrite(
            os.path.join(save_fol, os.path.basename(image_file).split('.')[0] + f'_masked.jpg'),
            face_mask[crop_start_y:crop_end_y, crop_start_x:crop_end_x, :],
        )
        shutil.copy(mat_file, save_fol)
        shutil.copy(mat_file, os.path.join(save_fol, os.path.basename(mat_file).split('.')[0] + f'_masked.mat'))



elps = time.time() - st
print('time used: %.0f m : %.0f s' % (elps // 60, elps % 60))

nondetected = pd.DataFrame(nondetected)
nondetected.to_csv(f'non_detected.csv', header=None, index=None)
morethanone = pd.DataFrame(morethanone)
morethanone.to_csv(f'multi_detected.csv', header=None, index=None)