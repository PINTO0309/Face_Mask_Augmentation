import os
import cv2
import glob
from tqdm import tqdm
from copy import deepcopy
from natsort import natsorted
from argparse import ArgumentParser
from ibug.face_detection import RetinaFacePredictor

face_detector = RetinaFacePredictor(
    threshold=0.8,
    device='cuda:0',
    model=RetinaFacePredictor.get_model('resnet50')
)

morethanone = []
nondetected = []

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-i',
        '--image_folder_path',
        type=str,
        default='300W_LP',
    )
    args = parser.parse_args()

    image_files = glob.glob(f"{args.image_folder_path}/*/*.jpg")

    image_count = 0
    for image_file in tqdm(natsorted(image_files), dynamic_ncols=True):

        dirname = os.path.dirname(image_file)
        # print(f'@@@ dirname: {dirname} split: {dirname.split("/")}')
        new_dirname = f'{args.image_folder_path}_onlyone_person/{dirname.split("/")[1]}'
        os.makedirs(new_dirname, exist_ok=True)

        image = cv2.imread(image_file)

        debug_image = deepcopy(image)
        debug_image = debug_image[..., ::-1]

        detected_faces = face_detector(debug_image, rgb=True)

        if len(detected_faces) == 1:

            # for face_box in detected_faces:
            #     cv2.rectangle(
            #         image,
            #         (int(face_box[0]), int(face_box[1])),
            #         (int(face_box[2]), int(face_box[3])),
            #         (255,255,255),
            #         2,
            #     )
            #     cv2.rectangle(
            #         image,
            #         (int(face_box[0]), int(face_box[1])),
            #         (int(face_box[2]), int(face_box[3])),
            #         (0,255,0),
            #         1,
            #     )
            #     cv2.putText(
            #         image,
            #         f'{face_box[4]:.2f}',
            #         (
            #             int(face_box[0]),
            #             int(face_box[1]-10) if face_box[1]-10 > 0 else 20
            #         ),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.7,
            #         (255, 255, 255),
            #         2,
            #         cv2.LINE_AA,
            #     )
            #     cv2.putText(
            #         image,
            #         f'{face_box[4]:.2f}',
            #         (
            #             int(face_box[0]),
            #             int(face_box[1]-10) if face_box[1]-10 > 0 else 20
            #         ),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.7,
            #         (0, 255, 0),
            #         1,
            #         cv2.LINE_AA,
            #     )

            # cv2.imshow("test", image)

            # key = cv2.waitKey(0)
            # if key == 27: # ESC
            #     break

            basename = os.path.basename(image_file)
            cv2.imwrite(f'{new_dirname}/{basename}', image)
            image_count += 1

    print(f'image_count: {image_count}')

if __name__ == "__main__":
    main()