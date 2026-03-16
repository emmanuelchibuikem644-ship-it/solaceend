import torch
import json
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification

MODEL_PATH = "emotion_model"

tokenizer = None
model = None

# load emotion mapping
with open("config/emotion_mapping.json") as f:
    emotion_map = json.load(f)


def load_model():
    global tokenizer, model

    if model is None:
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()

    return tokenizer, model


def predict_emotion(text):

    tokenizer, model = load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)

    emotion_id = torch.argmax(probs).item()

    emotion = emotion_map[str(emotion_id)]

    return emotion