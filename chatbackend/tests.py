import requests
from zipfile import ZipFile
import os

url = "https://drive.google.com/uc?export=download&id=1fJQfDT0TSIHVp7VqzzfpeAmWGHrV2aiR"

zip_path = "emotion_model.zip"

print("Downloading model...")

r = requests.get(url, stream=True)

with open(zip_path, "wb") as f:
    for chunk in r.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)

print("Download complete")

print("Extracting model...")

with ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall()

os.remove(zip_path)

print("Model ready!")