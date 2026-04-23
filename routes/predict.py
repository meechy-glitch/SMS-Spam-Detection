import torch
import torch.nn.functional as F
from fastapi import APIRouter, HTTPException
# from fastapi import Depends
# from sqlalchemy.orm import Session
# from typing import List

# from database.database import get_db
# from database.models import Prediction
from schemas import SMSRequest, PredictionResponse
# from schemas import PredictionRecord
from ml.model import tokenizer, model

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(body: SMSRequest):  # add: db: Session = Depends(get_db)
    sms_input = body.smsInput.strip()

    if not sms_input:
        raise HTTPException(status_code=400, detail="No message provided")

    inputs = tokenizer(
        sms_input, return_tensors="pt", truncation=True, max_length=128, padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]

    ham_prob = round(probs[0].item() * 100, 1)
    spam_prob = round(probs[1].item() * 100, 1)
    is_spam = spam_prob > 50

    # record = Prediction(
    #     message=sms_input,
    #     result="Spam" if is_spam else "Not Spam",
    #     confidence=spam_prob if is_spam else ham_prob,
    #     spam_probability=spam_prob,
    #     ham_probability=ham_prob,
    # )
    # db.add(record)
    # db.commit()
    # db.refresh(record)
    # return record

    return {
        "result": "Spam" if is_spam else "Not Spam",
        "confidence": spam_prob if is_spam else ham_prob,
        "spam_probability": spam_prob,
        "ham_probability": ham_prob,
    }


# @router.get("/predictions", response_model=List[PredictionRecord])
# def get_predictions(db: Session = Depends(get_db)):
#     return db.query(Prediction).order_by(Prediction.id.desc()).all()
