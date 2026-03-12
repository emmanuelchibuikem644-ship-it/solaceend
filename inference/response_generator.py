from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/DialoGPT-medium"
)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/DialoGPT-medium"
)


def generate_response(message):

    inputs = tokenizer.encode(
        message + tokenizer.eos_token,
        return_tensors="pt"
    )

    output = model.generate(
        inputs,
        max_length=100,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(
        output[:, inputs.shape[-1]:][0],
        skip_special_tokens=True
    )

    return response