import base64
import binascii

from fastapi import FastAPI, HTTPException

from results import ResultsJsonEncoder
from reticulai import ReticulAI

app = FastAPI()


@app.post("/detect")
async def detect(image_data: str) -> list[dict]:
    detector = ReticulAI()

    try:
        _ = base64.b64decode(image_data, validate=True)
    except binascii.Error as e:
        raise HTTPException(status_code=400, detail=f"{e} Invalid image data")

    result = detector.detect(image_data)

    return ResultsJsonEncoder.response_with_crops(result, detector.names)
