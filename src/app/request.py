import os
import sys

import requests

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

from src import utils

image_path = "data/TEST/test_phone/0001.jpg"
api_host = "http://localhost:8000/"
request_url = f"{api_host}/detect"

data = utils.open_file_as_b64(image_path)
response = requests.post(f"{api_host}/detect", json={"data": data})

print(response.text)

response = requests.post(f"{api_host}/detect-crops", json={"data": data})

print(response.text)

response = requests.post(f"{api_host}/detect-count", json={"data": data})

print(response.text)
