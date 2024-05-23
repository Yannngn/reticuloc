import os
import sys

from fastapi import FastAPI

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

from src import utils
from src.app.schemas.detection import DetectionBase
from src.app.schemas.detection_count import DetectionCountBase
from src.app.schemas.detection_crops import DetectionCropsBase
from src.app.schemas.image import ImageBase
from src.model import json_encoder
from src.model.reticulai import ReticulAI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/detect")
async def detect(image_data: ImageBase) -> list[DetectionBase]:
    _ = utils.check_b64(image_data.data)

    detector = ReticulAI()

    result = detector.detect(image_data.data)

    response = json_encoder.response_with_boxes(result, detector.names)

    return response


@app.post("/detect-crops")
async def detect_crops(image_data: ImageBase) -> list[DetectionCropsBase]:
    _ = utils.check_b64(image_data.data)

    detector = ReticulAI()

    result = detector.detect(image_data.data)

    response = json_encoder.response_with_crops(result, detector.names)

    return response


@app.post("/detect-count")
async def detect_count(image_data: ImageBase) -> DetectionCountBase:
    _ = utils.check_b64(image_data.data)

    detector = ReticulAI()

    result = detector.detect(image_data.data)

    response = json_encoder.response_with_count(result, detector.names)

    return response
