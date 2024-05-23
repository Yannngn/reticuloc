from typing import Any

import numpy as np
import torch
from ultralytics.engine.results import Boxes, Results


def boxes_annotation(result: Results) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = result.boxes

    assert isinstance(boxes, Boxes)

    xyxyn = boxes.xyxyn.cpu().numpy() if isinstance(boxes.xyxyn, torch.Tensor) else boxes.xyxyn
    labels = boxes.cls.int().cpu().numpy() if isinstance(boxes.cls, torch.Tensor) else boxes.cls
    conf = boxes.conf.cpu().numpy() if isinstance(boxes.conf, torch.Tensor) else boxes.conf

    return xyxyn, labels, conf


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

    xyxyn = boxes.xyxyn.cpu().numpy() if isinstance(boxes.xyxyn, torch.Tensor) else boxes.xyxyn

    return cropped_images, xyxyn, labels, conf


def get_counts(result: Results) -> dict[int, dict[str, Any]]:
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

    count_dict: dict[int, dict[str, Any]] = {}
    for label in conf_mean:
        count_dict[label] = {"count": len(conf_mean[label]), "conf": np.mean(conf_mean[label]).item()}

    return count_dict
