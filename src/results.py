import base64
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Boxes, Results


class ResultsProcessor:
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

    @staticmethod
    def get_counts(result: Results) -> dict[int, dict[str, float | int]]:
        boxes = result.boxes

        assert isinstance(boxes, Boxes)

        labels = boxes.cls.int().cpu().numpy() if isinstance(boxes.cls, torch.Tensor) else boxes.cls
        conf = boxes.conf.cpu().numpy() if isinstance(boxes.conf, torch.Tensor) else boxes.conf

        conf_mean: dict[int, list[float]] = {}
        for label, conf in zip(labels, conf):
            if label not in conf_mean:
                conf_mean[label] = [conf]
            else:
                conf_mean[label].append(conf)

        count_dict: dict[int, dict[str, float | int]] = {}
        for label in conf_mean:
            count_dict[label] = {"count": len(conf_mean[label]), "conf": np.mean(conf_mean[label]).item()}

        return count_dict


class ResultsJsonEncoder:
    @staticmethod
    def response_with_crops(result: Results, label_map: dict[int, str] | None = None) -> list[dict[str, Any]]:
        crops = ResultsProcessor.cut_boxes(result)

        response = [
            {
                "image": ResultsJsonEncoder.image_to_bytes(image),
                "label": f"{label_map[label] if isinstance(label_map, dict) else label}",
                "conf": float(conf),
                "box": box.tolist(),
            }
            for (image, label, conf, box) in zip(*crops)
        ]

        return response

    @staticmethod
    def response_with_boxes(result: Results, label_map: dict[int, str] | None = None) -> list[dict[str, Any]]:
        annotations = ResultsProcessor.boxes_annotation(result)

        response = [
            {
                "label": f"{label_map[label] if isinstance(label_map, dict) else label}",
                "conf": float(conf),
                "box": box.tolist(),
            }
            for box, label, conf in zip(*annotations)
        ]

        return response

    @staticmethod
    def response_with_count(result: Results, label_map: dict[int, str] | None = None) -> dict[str, float | int]:
        dict_ = ResultsProcessor.get_counts(result)

        response = {f"{label_map[label] if isinstance(label_map, dict) else label}": v for label, v in dict_.items()}

        response = {f"{label}_{metric}": vv for label, v in response.items() for metric, vv in v.items()}

        return response

    @staticmethod
    def image_to_bytes(image: np.ndarray) -> str:
        image_bytes = cv2.imencode(".jpg", image)[1].tobytes()
        encoded_image = base64.urlsafe_b64encode(image_bytes).decode()

        return encoded_image
