import os
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model/distilbert-spam/final"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at '{MODEL_PATH}'. Run `python train.py` first."
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    sms_input = data.get("smsInput", "").strip()

    if not sms_input:
        return jsonify({"error": "No message provided"}), 400

    inputs = tokenizer(
        sms_input, return_tensors="pt", truncation=True, max_length=128, padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]

    ham_prob = round(probs[0].item() * 100, 1)
    spam_prob = round(probs[1].item() * 100, 1)
    is_spam = spam_prob > 50

    return jsonify(
        {
            "result": "Spam" if is_spam else "Not Spam",
            "confidence": spam_prob if is_spam else ham_prob,
            "spam_probability": spam_prob,
            "ham_probability": ham_prob,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
