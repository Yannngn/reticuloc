import os
import sys

from ultralytics import YOLO
from ultralytics.engine.results import Results

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

from src import utils


class ReticulAI:
    # checkpoint_name = "yolov9c_aug.pt"
    checkpoint_name = "yolov8n_aug.pt"

    def __init__(self, checkpoints_dir: str = "checkpoints"):
        self.checkpoints_dir = checkpoints_dir

        self._load_model()

    def detect(self, image_data: str) -> Results:
        return self.model.predict(utils.write_image_from_b64(image_data), conf=0.75, verbose=False)[0]

    def _load_model(self):
        self.model: YOLO = YOLO(os.path.join(self.checkpoints_dir, self.checkpoint_name))
        self.names: dict[int, str] = self.model.names  # type: ignore
