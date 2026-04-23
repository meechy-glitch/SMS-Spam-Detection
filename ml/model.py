import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "model/distilbert-spam/final"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at '{MODEL_PATH}'. Run `python ml/train.py` first."
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
