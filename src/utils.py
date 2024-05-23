import base64
import tempfile

import cv2
import numpy as np


def check_b64(data: str) -> None:
    _ = base64.urlsafe_b64decode(data)


def open_file_as_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.urlsafe_b64encode(f.read()).decode()


def write_image_from_b64(data: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(base64.urlsafe_b64decode(data))

        return f.name


def image_to_b64(image: np.ndarray) -> str:
    image_bytes = cv2.imencode(".jpg", image)[1].tobytes()
    encoded_image = base64.urlsafe_b64encode(image_bytes).decode()

    return encoded_image
