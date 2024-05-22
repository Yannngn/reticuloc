import base64
import os

import cv2
import numpy as np
import requests

image_path = "data/TEST/test_phone/0000.jpg"
api_host = "http://localhost:8000"
request_url = f"{api_host}/detect"

with open(image_path, "rb") as f:
    fmt = os.path.splitext(image_path)[1].replace(".", "")
    data = {"image": f"data:image/{fmt};base64,{base64.urlsafe_b64encode(f.read()).decode()}"}

response = requests.post(request_url, json=data)

for annotation in eval(response.text):
    image = base64.urlsafe_b64decode(annotation["image"])
    cv2.imshow(annotation["label"], cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR))

    cv2.waitKey(0)
