from pathlib import Path
import requests
from zipfile import ZipFile
from django.apps import AppConfig
import os

class ChatbackendConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'chatbackend'

    def ready(self):
        base_dir = Path(__file__).resolve().parent.parent
        model_path = base_dir / "emotion_model"

        if not model_path.exists():
            print("Downloading emotion model...")

            url = "https://drive.google.com/uc?export=download&id=1fJQfDT0TSIHVp7VqzzfpeAmWGHrV2aiR"
            zip_path = base_dir / "emotion_model.zip"

            r = requests.get(url, stream=True)
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            with ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(base_dir)

            os.remove(zip_path)
            print("Emotion model downloaded successfully.")