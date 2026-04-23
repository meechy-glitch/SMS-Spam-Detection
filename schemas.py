from pydantic import BaseModel
from datetime import datetime


class SMSRequest(BaseModel):
    smsInput: str


class PredictionResponse(BaseModel):
    result: str
    confidence: float
    spam_probability: float
    ham_probability: float

    model_config = {"from_attributes": True}


class PredictionRecord(PredictionResponse):
    id: int
    message: str
    created_at: datetime

    model_config = {"from_attributes": True}
