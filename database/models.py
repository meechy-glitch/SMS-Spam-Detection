from sqlalchemy import Column, Integer, String, Float, DateTime, func
from .database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    message = Column(String, nullable=False)
    result = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    spam_probability = Column(Float, nullable=False)
    ham_probability = Column(Float, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
