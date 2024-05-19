import base64
import binascii

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from results import ResultsJsonEncoder
from reticulai import ReticulAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class ImageData(BaseModel):
    image_data: str


@app.post("/detect")
async def detect(image_data: ImageData) -> list[dict]:
    detector = ReticulAI()

    try:
        _ = base64.b64decode(image_data.image_data, validate=True)
    except binascii.Error as e:
        raise HTTPException(status_code=400, detail=f"{e} Invalid image data")

    result = detector.detect(image_data.image_data)

    return ResultsJsonEncoder.response_with_boxes(result, detector.names)
