from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from pathlib import Path
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Get project base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Correct model path using pathlib
MODEL_PATH = BASE_DIR / "emotion_model"

# Load tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

# Load emotion mapping
with open(BASE_DIR / "config" / "emotion_mapping.json") as f:
    emotion_map = json.load(f)


def predict_emotion(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return emotion_map[str(predicted_class)]


@csrf_exempt
def chat_api(request):

    if request.method == "POST":

        data = json.loads(request.body)

        message = data.get("message")

        emotion = predict_emotion(message)

        return JsonResponse({
            "emotion": emotion,
            "reply": f"I sense you may be feeling {emotion}. I'm here to listen."
        })

    return JsonResponse({"message": "API is running"})