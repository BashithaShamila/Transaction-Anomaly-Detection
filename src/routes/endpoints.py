from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
from src.model.fraud_model import predict_anomaly

router = APIRouter()


@router.get("/ping")
def ping():
    return {"message": "pong"}


class Transaction(BaseModel):
    """Transaction data model for anomaly detection"""
    features: List[float] = Field(..., description="List of transaction features for the model")
    transaction_id: Optional[str] = Field(None, description="Optional transaction identifier")


class TransactionBatch(BaseModel):
    """Batch of transactions for anomaly detection"""
    transactions: List[Transaction] = Field(..., description="List of transactions to analyze")


@router.post("/predict", response_model=Dict[str, Any])
def detect_anomalies(transaction_data: TransactionBatch):
    """
    Detect anomalies in transaction data using Isolation Forest model
    """
    try:
        # Extract features from all transactions
        features = np.array([t.features for t in transaction_data.transactions])
        
        # Run anomaly detection
        results = predict_anomaly(features)
        
        # Add transaction IDs to the results if provided
        transaction_ids = [t.transaction_id for t in transaction_data.transactions]
        
        # Format response
        response = {
            "results": [
                {
                    "transaction_id": tid if tid else f"transaction_{i}",
                    "anomaly_score": float(results["scores"][i]),
                    "is_anomaly": bool(results["predictions"][i] == 1)
                }
                for i, tid in enumerate(transaction_ids)
            ],
            "summary": {
                "total_transactions": len(transaction_data.transactions),
                "anomalies_detected": sum(results["predictions"]),
                "detection_threshold": -0.5
            }
        }
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing transaction data: {str(e)}")


def setup_routes(app):
    app.include_router(router)
