import os
import sys

import numpy as np
from ultralytics.engine.results import Results

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

from src import utils
from src.app.schemas.detection import DetectionBase
from src.app.schemas.detection_count import DetectionCountBase
from src.app.schemas.detection_crops import DetectionCropsBase
from src.model import postprocessing
from src.model import utils as mu


def response_with_crops(result: Results, label_map: dict[int, str] | None = None) -> list[DetectionCropsBase]:
    crops = postprocessing.cut_boxes(result)

    # response = [
    #     {
    #         "data": utils.image_to_base64(image),
    #         "box": box.tolist() if isinstance(box, np.ndarray) else box,
    #         "label": f"{label_map[label] if isinstance(label_map, dict) else label}",
    #         "conf": float(conf),
    #     }
    #     for (image, box, label, conf) in zip(*crops)
    # ]

    response = [
        DetectionCropsBase(
            data=utils.image_to_b64(image),
            box=box.tolist() if isinstance(box, np.ndarray) else box,
            label=mu.decode_label(label, label_map),
            conf=float(conf),
        )
        for image, box, label, conf in zip(*crops)
    ]

    return response


def response_with_boxes(result: Results, label_map: dict[int, str] | None = None) -> list[DetectionBase]:
    annotations = postprocessing.boxes_annotation(result)

    # response = [
    #     {
    #         "box": box.tolist() if isinstance(box, np.ndarray) else box,
    #         "label": utils.decode_label(label, label_map),
    #         "conf": float(conf),
    #     }
    #     for box, label, conf in zip(*annotations)
    # ]

    response = [
        DetectionBase(
            box=box.tolist() if isinstance(box, np.ndarray) else box,
            label=mu.decode_label(int(label), label_map),
            conf=float(conf),
        )
        for box, label, conf in zip(*annotations)
    ]

    return response


def response_with_count(result: Results, label_map: dict[int, str] | None = None) -> DetectionCountBase:
    dict_ = postprocessing.get_counts(result)

    response = {mu.decode_label(int(label), label_map): v for label, v in dict_.items()}

    response = {f"{label}_{metric}": vv for label, v in response.items() for metric, vv in v.items()}

    response = DetectionCountBase(**response)

    return response
