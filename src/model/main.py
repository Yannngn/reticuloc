import os
import sys

import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

from src import utils
from src.model import json_encoder, postprocessing
from src.model.reticulai import ReticulAI


def main():
    show = 0
    detector = ReticulAI()

    print(detector.names)

    image_path = "data/TEST/test_phone/0000.jpg"

    image_data = utils.open_file_as_b64(image_path)

    result = detector.detect(image_data)

    if show:
        cv2.imshow(f"input", cv2.imread(image_path))
        result.show()

        crops = postprocessing.cut_boxes(result)

        for image, _, label, conf in zip(*crops):
            cv2.imshow(f"{detector.names[label]} - {conf:.3%}", image)

        cv2.waitKey(0)

    response = json_encoder.response_with_crops(result, detector.names)

    print(response[0])


if __name__ == "__main__":
    main()
