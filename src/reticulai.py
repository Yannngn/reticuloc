import base64
import os
import sys
import tempfile

import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from results import ResultsJsonEncoder, ResultsProcessor


class ReticulAI:
    # checkpoint_name = "yolov9c_aug.pt"
    checkpoint_name = "yolov8n_aug.pt"

    def __init__(self, checkpoints_dir: str = "checkpoints"):
        self.checkpoints_dir = checkpoints_dir

        self._load_model()

    def detect(self, image_data: str, extension: str) -> Results:

        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as f:
            f.write(base64.urlsafe_b64decode(image_data))

            return self.model.predict(f.name, conf=0.75, verbose=False)[0]

    def _load_model(self):
        self.model: YOLO = YOLO(os.path.join(self.checkpoints_dir, self.checkpoint_name))
        self.names: dict[int, str] = self.model.names  # type: ignore


def main():
    show = False
    detector = ReticulAI()

    image_path = "data/TEST/test_phone/0000.jpg"

    with open(image_path, "rb") as f:
        image_data = base64.urlsafe_b64encode(f.read()).decode()

    result = detector.detect(image_data, os.path.splitext(image_path)[1])

    if show:
        cv2.imshow(f"input", cv2.imread(image_path))
        result.show()

        crops = ResultsProcessor.cut_boxes(result)

        for image, label, conf, _ in zip(*crops):
            cv2.imshow(f"{detector.names[label]} - {conf:.3%}", image)

        cv2.waitKey(0)

    response = ResultsJsonEncoder.response_with_crops(result, detector.names)

    print(print([type(v) for v in response[0].values()]))


if __name__ == "__main__":
    main()
