from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

MODEL_PATH = "emotion_model"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

with open("config/emotion_mapping.json") as f:
    emotion_map = json.load(f)


def predict_emotion(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    outputs = model(**inputs)
    logits = outputs.logits

    predicted_class = torch.argmax(logits, dim=1).item()

    emotion = emotion_map[str(predicted_class)]

    return emotion


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