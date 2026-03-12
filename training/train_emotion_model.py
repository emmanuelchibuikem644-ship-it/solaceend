from compute_metrics import compute_metrics
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import json

# ----------------------------
# Load custom labels
# ----------------------------
with open("config/labels.json") as f:
    label_config = json.load(f)

id2label = {int(k): v for k, v in label_config["id2label"].items()}
label2id = label_config["label2id"]
num_labels = len(label2id)

# ----------------------------
# Load emotion mapping
# ----------------------------
with open("config/emotion_mapping.json") as f:
    emotion_mapping = json.load(f)

# ----------------------------
# Load go_emotions dataset
# ----------------------------
dataset = load_dataset("go_emotions")
train_dataset = dataset["train"]
val_dataset = dataset["validation"]

# ----------------------------
# Map go_emotions → custom labels
# ----------------------------
def map_labels(example):
    original_labels = example["labels"]

    for label_id in original_labels:
        label_name = dataset["train"].features["labels"].feature.names[label_id]

        if label_name in emotion_mapping:
            mapped_label = emotion_mapping[label_name]
            example["labels"] = label2id[mapped_label]
            return example

    example["labels"] = label2id["neutral"]
    return example


train_dataset = train_dataset.map(map_labels)
val_dataset = val_dataset.map(map_labels)

# ----------------------------
# Load tokenizer
# ----------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# ----------------------------
# Tokenize function
# ----------------------------
def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

# Apply tokenization
train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

# ----------------------------
# Set dataset format for PyTorch
# ----------------------------
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

print(f"Number of classes: {num_labels}")
print(f"Custom labels: {list(label2id.keys())}")

# ----------------------------
# Load model
# ----------------------------
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# ----------------------------
# Training arguments
# ----------------------------
training_args = TrainingArguments(
    output_dir="emotion_model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="logs",
    logging_steps=50,
)

# ----------------------------
# Trainer with compute_metrics
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# ----------------------------
# Train
# ----------------------------
trainer.train()

# ----------------------------
# Save model & tokenizer
# ----------------------------
model.save_pretrained("emotion_model")
tokenizer.save_pretrained("emotion_model")

print("Training finished successfully.")