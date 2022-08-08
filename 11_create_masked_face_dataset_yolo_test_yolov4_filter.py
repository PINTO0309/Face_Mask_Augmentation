#!/usr/bin/env python

import os
import cv2
import glob
import copy
import numpy as np
import onnxruntime
from tqdm import tqdm
from natsort import natsorted
from argparse import ArgumentParser
from typing import Tuple, Optional, List


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


def main():
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
        default='300W_LP_onlyone_person',
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

    image_files = glob.glob(f"{args.image_folder_path}/*/*.jpg")

    image_count = 0
    for image_file in tqdm(natsorted(image_files), dynamic_ncols=True):

        dirname = os.path.dirname(image_file)
        # print(f'@@@ dirname: {dirname} split: {dirname.split("/")}')
        new_dirname = f'{args.image_folder_path}_yolov4_filterd/{dirname.split("/")[1]}'
        os.makedirs(new_dirname, exist_ok=True)

        image = cv2.imread(image_file)

        debug_image = copy.deepcopy(image)
        face_boxes, face_scores = model(debug_image)

        if len(face_boxes) == 1:

            # for face_box, face_score in zip(face_boxes, face_scores):

            #     x_min = int(face_box[0])
            #     y_min = int(face_box[1])
            #     x_max = int(face_box[2])
            #     y_max = int(face_box[3])

            #     # add margin
            #     y_min = int(max(0, y_min - abs(y_min - y_max) / 17))
            #     y_max = int(min(image.shape[0], y_max + abs(y_min - y_max) / 17))
            #     x_min = int(max(0, x_min - abs(x_min - x_max) / 7))
            #     x_max = min(image.shape[1], x_max + abs(x_min - x_max) / 7)
            #     x_max = int(min(x_max, image.shape[1]))

            #     cv2.rectangle(
            #         debug_image,
            #         (x_min, y_min),
            #         (x_max, y_max),
            #         (255,255,255),
            #         2,
            #     )
            #     cv2.rectangle(
            #         debug_image,
            #         (x_min, y_min),
            #         (x_max, y_max),
            #         (0,255,0),
            #         1,
            #     )
            #     cv2.putText(
            #         debug_image,
            #         f'{face_score[0]:.2f}',
            #         (
            #             x_min,
            #             y_min-10 if y_min-10 > 0 else 20
            #         ),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.7,
            #         (255, 255, 255),
            #         2,
            #         cv2.LINE_AA,
            #     )
            #     cv2.putText(
            #         debug_image,
            #         f'{face_score[0]:.2f}',
            #         (
            #             x_min,
            #             y_min-10 if y_min-10 > 0 else 20
            #         ),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.7,
            #         (0, 255, 0),
            #         1,
            #         cv2.LINE_AA,
            #     )

            # cv2.imshow("test", debug_image)

            # key = cv2.waitKey(0)
            # if key == 27: # ESC
            #     break

            basename = os.path.basename(image_file)
            cv2.imwrite(f'{new_dirname}/{basename}', image)
            image_count += 1

    print(f'image_count: {image_count}')

if __name__ == "__main__":
    main()