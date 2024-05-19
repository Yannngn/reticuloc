import base64

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Boxes, Results


class ResultsCutter:
    @staticmethod
    def boxes_annotation(result: Results) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        boxes = result.boxes

        assert isinstance(boxes, Boxes)

        xyxy = boxes.xyxy.cpu().numpy() if isinstance(boxes.xyxy, torch.Tensor) else boxes.xyxy
        labels = boxes.cls.int().cpu().numpy() if isinstance(boxes.cls, torch.Tensor) else boxes.cls
        conf = boxes.conf.cpu().numpy() if isinstance(boxes.conf, torch.Tensor) else boxes.conf

        return xyxy, labels, conf

    @staticmethod
    def cut_boxes(result: Results) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        image = result.orig_img

        boxes = result.boxes

        assert isinstance(boxes, Boxes)

        labels = boxes.cls.int().cpu().numpy() if isinstance(boxes.cls, torch.Tensor) else boxes.cls
        conf = boxes.conf.cpu().numpy() if isinstance(boxes.conf, torch.Tensor) else boxes.conf
        xyxy = boxes.xyxy.cpu().numpy() if isinstance(boxes.xyxy, torch.Tensor) else boxes.xyxy

        cropped_images = []
        for box in xyxy:
            x, y, x1, y1 = map(round, box)
            cropped_images.append(image[y:y1, x:x1])

        return cropped_images, labels, conf, xyxy


class ResultsJsonEncoder:
    @staticmethod
    def response_with_crops(result: Results, label_map: dict | None = None) -> list[dict]:
        cuts = ResultsCutter.cut_boxes(result)

        response = [
            {
                "image": ResultsJsonEncoder.image_to_bytes(image),
                "label": label_map[label] if isinstance(label_map, dict) else label,
                "conf": conf,
                "box": list(box),
            }
            for (image, label, conf, box) in zip(*cuts)
        ]

        return response

    @staticmethod
    def response_with_boxes(result: Results, label_map: dict | None = None) -> list[dict]:
        annotations = ResultsCutter.boxes_annotation(result)

        response = [
            {
                "label": label_map[label] if isinstance(label_map, dict) else label,
                "conf": conf,
                "box": list(box),
            }
            for box, label, conf in zip(*annotations)
        ]

        return response

    @staticmethod
    def image_to_bytes(image: np.ndarray) -> str:
        image_bytes = cv2.imencode(".jpg", image)[1].tobytes()
        encoded_image = base64.b64encode(image_bytes).decode()

        return encoded_image
