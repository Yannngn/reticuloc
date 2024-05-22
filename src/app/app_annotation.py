import base64
import binascii

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas.image import ImageBase
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


@app.post("/detect")
async def detect(image_data: ImageBase) -> list[dict]:
    detector = ReticulAI()

    try:
        data = image_data.image
        extension = data.split(";")[0].split("/")[1]
        data = data.replace(f"data:image/{extension};base64,", "")

        _ = base64.urlsafe_b64decode(data)
    except binascii.Error:
        raise HTTPException(status_code=400, detail=f"Invalid image data")

    result = detector.detect(data, "." + extension)

    return ResultsJsonEncoder.response_with_boxes(result, detector.names)
