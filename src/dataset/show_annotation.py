import os
import sys

import cv2
from glob2 import glob

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

from src.dataset.labels import DICT

COLORS = {0: "#8B0000", 1: "#FFA500", 2: "#FA8072"}


def show_dir(yolo_dir: str = "data/yolo"):
    labels = glob(os.path.join(yolo_dir, "*.txt"))

    for label_file in labels:
        image_path = label_file.replace(".txt", ".jpg")
        image = cv2.imread(image_path)

        with open(label_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            bbox = map(float, line.split(" ")[1:])
            label = int(line.split(" ")[0])

            x, y, w, h = bbox

            x *= image.shape[1]
            y *= image.shape[0]
            w *= image.shape[1]
            h *= image.shape[0]

            x = x - w / 2
            y = y - h / 2

            x, y, w, h = map(int, (x, y, w, h))

            color = tuple(int(COLORS[label][i : i + 2], 16) for i in (1, 3, 5))

            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness=2)
            cv2.putText(image, DICT[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color=color, thickness=2)  # Draw the label

        cv2.imshow("Image with Annotations", image)
        cv2.waitKey(0)

    # Close all windows
    cv2.destroyAllWindows()
