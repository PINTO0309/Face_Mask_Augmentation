#!/usr/bin/env python

import os
import cv2
import glob
import copy
import json
import numpy as np
import onnxruntime
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from argparse import ArgumentParser
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split

# input image width/height of the yolov4 model, set by command-line argument
INPUT_WIDTH  = 0
INPUT_HEIGHT = 0

# Minimum width/height of objects for detection (don't learn from objects smaller than these)
MIN_W = 5
MIN_H = 5

# Do K-Means clustering in order to determine "anchor" sizes
DO_KMEANS = True
KMEANS_CLUSTERS = 9
BBOX_WHS = []  # keep track of bbox width/height with respect to 640x640


class YOLOv4ONNX(object):
    def __init__(
        self,
        model_path: Optional[str] = 'yolov4_headdetection_480x640_post.onnx',
        input_shape: Optional[Tuple[int,int]] = (480, 640),
        class_score_th: Optional[float] = 0.20,
        providers: Optional[List] = [
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        """YOLOv4ONNX

        Parameters
        ----------
        model_path: Optional[str]
            ONNX file path for YOLOv4

        input_shape: Optional[Tuple[int,int]]
            Model Input Resolution, Default: (480,640)

        class_score_th: Optional[float]

        class_score_th: Optional[float]
            Score threshold. Default: 0.20

        providers: Optional[List]
            Name of onnx execution providers
            Default:
            [
                'TensorrtExecutionProvider',
                'CUDAExecutionProvider',
                'CPUExecutionProvider',
            ]
        """
        # Input size
        self.input_shape = input_shape

        # Threshold
        self.class_score_th = class_score_th

        # Model loading
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 3
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_option,
            providers=providers,
        )
        self.providers = self.onnx_session.get_providers()

        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_name = self.onnx_session.get_outputs()[0].name


    def __call__(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """YOLOV4ONNX

        Parameters
        ----------
        image: np.ndarray
            Entire image

        Returns
        -------
        faceboxes: np.ndarray
            Predicted face boxes: [facecount, x1, y1, x2, y2]

        facescores: np.ndarray
            Predicted face box confs: [facecount, conf]
        """
        temp_image = copy.deepcopy(image)

        # PreProcess
        resized_image = self.__preprocess(
            temp_image,
        )

        # Inference
        inferece_image = np.asarray([resized_image], dtype=np.float32)
        boxes = self.onnx_session.run(
            None,
            {self.input_name: inferece_image},
        )[0]

        # PostProcess
        faceboxes, facescores = self.__postprocess(
            image= temp_image,
            boxes=boxes,
        )

        return faceboxes, facescores


    def __preprocess(
        self,
        image: np.ndarray,
        swap: Optional[Tuple[int,int,int]] = (2, 0, 1),
    ) -> np.ndarray:
        """__preprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image

        swap: tuple
            HWC to CHW: (2,0,1)
            CHW to HWC: (1,2,0)
            HWC to HWC: (0,1,2)
            CHW to CHW: (0,1,2)

        Returns
        -------
        resized_image: np.ndarray
            Resized and normalized image.
        """
        # Normalization + BGR->RGB
        resized_image = cv2.resize(
            image,
            (
                int(self.input_shape[1]), # type: ignore
                int(self.input_shape[0]), # type: ignore
            )
        )
        resized_image = np.divide(resized_image, 255.0) # type: ignore
        resized_image = resized_image.transpose(swap)
        resized_image = np.ascontiguousarray(resized_image, dtype=np.float32)
        return resized_image


    def __postprocess(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """__postprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image.

        boxes: np.ndarray
            (boxcount, 5) = (boxcount, x1y1x2y2score)

        Returns
        -------
        faceboxes: np.ndarray
            Predicted face boxes: [facecount, x1, y1, x2, y2]

        facescores: np.ndarray
            Predicted face box confs: [facecount, score]
        """
        image_height = image.shape[0]
        image_width = image.shape[1]

        scores = boxes[:,4]
        keep_idxs = scores > self.class_score_th
        boxes_keep = boxes[keep_idxs, :]

        faceboxes = []
        facescores = []

        if len(boxes_keep) > 0:
            boxes_keep[:, 0] = boxes_keep[:, 0] * image_width
            boxes_keep[:, 1] = boxes_keep[:, 1] * image_height
            boxes_keep[:, 2] = boxes_keep[:, 2] * image_width
            boxes_keep[:, 3] = boxes_keep[:, 3] * image_height

            for box in boxes_keep:
                x_min = int(box[0]) if int(box[0]) > 0 else 0
                y_min = int(box[1]) if int(box[1]) > 0 else 0
                x_max = int(box[2]) if int(box[2]) < image_width else image_width
                y_max = int(box[3]) if int(box[3]) < image_height else image_height
                score = box[4]

                faceboxes.append(
                    [
                        x_min,
                        y_min,
                        x_max,
                        y_max,
                    ]
                )
                facescores.append(
                    [
                        score
                    ]
                )

        return np.asarray(faceboxes), np.asarray(facescores) # type: ignore


class YOLOv7ONNX(object):
    def __init__(
        self,
        model_path: Optional[str] = 'yolov7_tiny_head_0.752_post_480x640.onnx',
        class_score_th: Optional[float] = 0.30,
        providers: Optional[List] = [
            # (
            #     'TensorrtExecutionProvider', {
            #         'trt_engine_cache_enable': True,
            #         'trt_engine_cache_path': '.',
            #         'trt_fp16_enable': True,
            #     }
            # ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        """YOLOv7ONNX
        Parameters
        ----------
        model_path: Optional[str]
            ONNX file path for YOLOv7
        class_score_th: Optional[float]
        class_score_th: Optional[float]
            Score threshold. Default: 0.30
        providers: Optional[List]
            Name of onnx execution providers
            Default:
            [
                (
                    'TensorrtExecutionProvider', {
                        'trt_engine_cache_enable': True,
                        'trt_engine_cache_path': '.',
                        'trt_fp16_enable': True,
                    }
                ),
                'CUDAExecutionProvider',
                'CPUExecutionProvider',
            ]
        """
        # Threshold
        self.class_score_th = class_score_th

        # Model loading
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 3
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_option,
            providers=providers,
        )
        self.providers = self.onnx_session.get_providers()

        self.input_shapes = [
            input.shape for input in self.onnx_session.get_inputs()
        ]
        self.input_names = [
            input.name for input in self.onnx_session.get_inputs()
        ]
        self.output_names = [
            output.name for output in self.onnx_session.get_outputs()
        ]


    def __call__(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """YOLOv7ONNX
        Parameters
        ----------
        image: np.ndarray
            Entire image
        Returns
        -------
        face_boxes: np.ndarray
            Predicted face boxes: [facecount, y1, x1, y2, x2]
        face_scores: np.ndarray
            Predicted face box scores: [facecount, score]
        """
        temp_image = copy.deepcopy(image)

        # PreProcess
        resized_image = self.__preprocess(
            temp_image,
        )

        # Inference
        inferece_image = np.asarray([resized_image], dtype=np.float32)
        scores, boxes = self.onnx_session.run(
            self.output_names,
            {input_name: inferece_image for input_name in self.input_names},
        )

        # PostProcess
        face_boxes, face_scores = self.__postprocess(
            image=temp_image,
            scores=scores,
            boxes=boxes,
        )

        return face_boxes, face_scores


    def __preprocess(
        self,
        image: np.ndarray,
        swap: Optional[Tuple[int,int,int]] = (2, 0, 1),
    ) -> np.ndarray:
        """__preprocess
        Parameters
        ----------
        image: np.ndarray
            Entire image
        swap: tuple
            HWC to CHW: (2,0,1)
            CHW to HWC: (1,2,0)
            HWC to HWC: (0,1,2)
            CHW to CHW: (0,1,2)
        Returns
        -------
        resized_image: np.ndarray
            Resized and normalized image.
        """
        # Normalization + BGR->RGB
        resized_image = cv2.resize(
            image,
            (
                int(self.input_shapes[0][3]),
                int(self.input_shapes[0][2]),
            )
        )
        resized_image = np.divide(resized_image, 255.0)
        resized_image = resized_image[..., ::-1]
        resized_image = resized_image.transpose(swap)
        resized_image = np.ascontiguousarray(
            resized_image,
            dtype=np.float32,
        )
        return resized_image


    def __postprocess(
        self,
        image: np.ndarray,
        scores: np.ndarray,
        boxes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """__postprocess
        Parameters
        ----------
        image: np.ndarray
            Entire image.
        scores: np.ndarray
            float32[N, 1]
        boxes: np.ndarray
            int64[N, 6]
        Returns
        -------
        faceboxes: np.ndarray
            Predicted face boxes: [facecount, y1, x1, y2, x2]
        facescores: np.ndarray
            Predicted face box confs: [facecount, score]
        """
        image_height = image.shape[0]
        image_width = image.shape[1]

        """
        Head Detector is
            N -> Number of boxes detected
            batchno -> always 0: BatchNo.0
            classid -> always 0: "Head"
        scores: float32[N,1],
        batchno_classid_y1x1y2x2: int64[N,6],
        """
        scores = scores
        keep_idxs = scores[:, 0] > self.class_score_th
        scores_keep = scores[keep_idxs, :]
        boxes_keep = boxes[keep_idxs, :]
        faceboxes = []
        facescores = []

        if len(boxes_keep) > 0:
            for box, score in zip(boxes_keep, scores_keep):
                x_min = max(int(box[3]), 0)
                y_min = max(int(box[2]), 0)
                x_max = min(int(box[5]), image_width)
                y_max = min(int(box[4]), image_height)

                faceboxes.append(
                    [x_min, y_min, x_max, y_max]
                )
                facescores.append(
                    score
                )

        return np.asarray(faceboxes), np.asarray(facescores)


def txt_line(cls, bbox, img_w, img_h):
    """Generate 1 line in the txt file."""
    x, y, w, h = bbox
    x = max(int(x), 0)
    y = max(int(y), 0)
    w = min(int(w), img_w - x)
    h = min(int(h), img_h - y)
    w_rescaled = float(w) * INPUT_WIDTH  / img_w
    h_rescaled = float(h) * INPUT_HEIGHT / img_h
    if w_rescaled < MIN_W or h_rescaled < MIN_H:
        return ''
    else:
        if DO_KMEANS:
            global BBOX_WHS
            BBOX_WHS.append((w_rescaled, h_rescaled))
        cx = (x + w / 2.) / img_w
        cy = (y + h / 2.) / img_h
        nw = float(w) / img_w
        nh = float(h) / img_h
        return f'{int(cls)} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n'


def process(set_, data_list, output_dir, model):
    """Process either 'train' or 'test' set."""
    jpgs = []
    raw_anno_count = 0
    print(f'** Processing Sets: {set_}')
    for image_file_path in tqdm(data_list, dynamic_ncols=True):
        image = cv2.imread(image_file_path)
        img_h, img_w, img_c = image.shape
        basename = os.path.basename(image_file_path)
        basename_without_ext = os.path.splitext(basename)[0]
        txt_path = output_dir / (f'{basename_without_ext}.txt')

        # inference
        face_boxes, face_scores = model(image)
        if len(face_boxes) == 1:
            line_count = 0
            with open(txt_path.as_posix(), 'w') as ftxt:
                for face_box, face_score in zip(face_boxes, face_scores):
                    x_min = int(face_box[0])
                    y_min = int(face_box[1])
                    x_max = int(face_box[2])
                    y_max = int(face_box[3])

                    # add margin
                    y_min = int(max(0, y_min - abs(y_min - y_max) / 17))
                    y_max = int(min(img_h, y_max + abs(y_min - y_max) / 17))
                    x_min = int(max(0, x_min - abs(x_min - x_max) / 7))
                    x_max = min(img_w, x_max + abs(x_min - x_max) / 7)
                    x_max = int(min(x_max, img_w))
                    w = int(x_max - x_min)
                    h = int(y_max - y_min)
                    bbox = [x_min, y_min, w, h]

                    line = txt_line(0, bbox, img_w, img_h)
                    if line:
                        ftxt.write(line)
                        line_count += 1

            if line_count > 0:
                jpgs.append(f'{output_dir}/{basename_without_ext}.jpg')
                cv2.imwrite(f'{output_dir}/{basename_without_ext}.jpg', image)
                raw_anno_count += 1


    print(f'** Processed Images: {raw_anno_count}')
    # write the 'data/300wlp-{args.dim}/train.txt' or 'data/300wlp-{args.dim}/test.txt'
    set_path = output_dir / (f'{set_}.txt')
    with open(set_path.as_posix(), 'w') as fset:
        for jpg in jpgs:
            fset.write(f'{jpg}\n')


def rm_txts(output_dir):
    """Remove txt files in output_dir."""
    for txt in output_dir.glob('*.txt'):
        if txt.is_file():
            txt.unlink()


def main():
    global INPUT_WIDTH, INPUT_HEIGHT

    parser = ArgumentParser()
    parser.add_argument(
        '-y',
        '--yolo_mode',
        type=str,
        default='yolov4',
        choices=['yolov4', 'yolov7']
    )
    parser.add_argument(
        '-i',
        '--image_folder_path',
        type=str,
        default='300W_LP_onlyone_person_yolov4_filterd',
    )
    parser.add_argument(
        '-d',
        '--dim',
        type=str,
        default='640x480',
        help='input width and height, e.g. 640x480'
    )
    args = parser.parse_args()

    yolo_mode = args.yolo_mode

    model = None
    if yolo_mode == 'yolov4':
        model = YOLOv4ONNX(
            model_path='yolov4_headdetection_480x640_post.onnx',
            class_score_th=0.80,
        )
    elif yolo_mode == 'yolov7':
        model = YOLOv7ONNX(
            model_path='yolov7_tiny_head_0.752_post_480x640.onnx',
            # class_score_th=0.90,
        )

    dim_split = args.dim.split('x')
    if len(dim_split) != 2:
        raise SystemExit(f'ERROR: bad spec of input dim ({args.dim})')
    INPUT_WIDTH, INPUT_HEIGHT = int(dim_split[0]), int(dim_split[1])
    if INPUT_WIDTH % 32 != 0 or INPUT_HEIGHT % 32 != 0:
        raise SystemExit(f'ERROR: bad spec of input dim ({args.dim})')

    output_dir = Path(f'data/300wlp-{args.dim}')
    output_dir.mkdir(parents=True, exist_ok=True)
    rm_txts(output_dir)

    # Train:Test = 0.95:0.05
    image_files = natsorted(glob.glob(f"{args.image_folder_path}/*/*.jpg"))
    train_list, test_list = train_test_split(
        image_files,
        test_size=0.05,
        train_size=0.95,
        random_state=1,
    )


    process(set_='test', data_list=test_list, output_dir=output_dir, model=model)
    process(set_='train', data_list=train_list, output_dir=output_dir, model=model)


if __name__ == "__main__":
    main()