import os
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from glob2 import glob

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

from src.dataset.labels import REV_DICT


def image_preprocess(image_path: str) -> tuple[tuple[int, int, int, int], np.ndarray]:
    image = cv2.imread(image_path)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(image_gray, 25, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    x, y, w, h = cv2.boundingRect(contours[0])

    cropped_image = image[y : y + h, x : x + w]

    return (x, y, w, h), cropped_image


def label_preprocess(label_path: str, roi_bbox: tuple[int, int, int, int]) -> list:
    annotations = []
    x, y, w, h = roi_bbox

    tree = ET.parse(label_path)
    root = tree.getroot()

    for obj in root.iter("object"):
        name = obj.find("name").text  # type: ignore
        label = REV_DICT[name]  # type: ignore

        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)  # type: ignore
        ymin = int(bbox.find("ymin").text)  # type: ignore
        xmax = int(bbox.find("xmax").text)  # type: ignore
        ymax = int(bbox.find("ymax").text)  # type: ignore

        x_center = (0.5 * (xmin + xmax) - x) / w
        x_center = max(min(x_center, 1), 0)

        y_center = (0.5 * (ymin + ymax) - y) / h
        y_center = max(min(y_center, 1), 0)

        width = (xmax - xmin) / w
        width = max(min(width, 1), 0)

        height = (ymax - ymin) / h
        height = max(min(height, 1), 0)

        annotations.append(f"{label} {x_center} {y_center} {width} {height}\n")

    return annotations


def convert_data_to_yolo(data_dir: str = "data", output_dir: str = "data"):
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")
    output_dir = os.path.join(output_dir, "yolo")

    os.makedirs(output_dir, exist_ok=True)

    images = glob(os.path.join(images_dir, "*.jpg"))

    for image_path in images:
        roi_bbox, image = image_preprocess(image_path)

        cv2.imwrite(os.path.join(output_dir, os.path.basename(image_path)), image)

        label_path = image_path.replace(images_dir, labels_dir).replace(".jpg", ".xml")

        annotations = label_preprocess(label_path, roi_bbox)

        txt_path = os.path.join(output_dir, os.path.basename(label_path).replace(".xml", ".txt"))

        with open(txt_path, "w") as file:
            file.writelines(annotations)
