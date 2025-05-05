from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import logging
import json
from pathlib import Path
import joblib
import pandas as pd
import os
import joblib        # Make sure joblib is imported
import traceback     # Make sure traceback is imported

# Initialize FastAPI app
app = FastAPI()

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION WITH EXACT PATHS ---
MODEL_PATH = "/app/models/isolation_forest_model.pkl"
MAPPING_PATH = "/app/models/category_mappings.json"



def load_model():
    """Load the trained Isolation Forest model."""
    logger.info(f"Attempting to load model from path: '{MODEL_PATH}'")

    # 1. Check if file exists
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at specified path: '{MODEL_PATH}'")
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    loaded_model = None # Initialize variable
    try:
        # 2. Attempt to load the model using joblib
        logger.info(f"Calling joblib.load('{MODEL_PATH}')")
        loaded_model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded successfully. Object type: {type(loaded_model)}")

        # Optional check: Ensure joblib didn't return None (very unlikely but safe)
        if loaded_model is None:
            logger.error("CRITICAL: joblib.load returned None unexpectedly!")
            raise ValueError("Model loading resulted in None unexpectedly.")

    except Exception as e:
        # 3. Catch ANY exception during loading
        logger.error(f"Failed to load model from {MODEL_PATH}: {e}", exc_info=True)
        # Log the traceback for detailed debugging
        # logger.error("Traceback explicitly logged:\n%s", traceback.format_exc())
        # 4. IMPORTANT: Re-raise the exception to stop execution and report the loading error
        raise

    # 5. Return the successfully loaded model object
    logger.info(f"Returning loaded model object.")
    return loaded_model

def load_category_mappings():
    """Load categorical encoding mappings from JSON."""
    logger.info(f"Loading category mappings from {MAPPING_PATH}")
    
    if not os.path.exists(MAPPING_PATH):
        raise FileNotFoundError(f"Mapping file not found at {MAPPING_PATH}")
    
    try:
        with open(MAPPING_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load category mappings: {e}")
        raise

def preprocess_input(data, cat_mappings):
    """Preprocess the input data and encode categorical columns."""
    try:
        df = pd.DataFrame([data])  # Wrap raw input as a DataFrame
        
        for col, mapping in cat_mappings.items():
            if col in df:
                df[col] = df[col].map(mapping).fillna(-1).astype(int)
            else:
                df[col] = -1  # Default if column not found in input
        
        return df
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

# Define the Pydantic model for transaction data
class TransactionData(BaseModel):
    cc_num: str
    merchant: str
    category: str
    amt: float
    first: str
    last: str
    gender: str
    street: str
    city: str
    state: str
    zip: str
    lat: float
    long: float
    city_pop: int
    job: str
    dob: str
    unix_time: int
    merch_lat: float
    merch_long: float

# This is the model that matches what your API is expecting
class PredictRequest(BaseModel):
    transactions: TransactionData

@app.post("/predict")
async def predict_anomaly(request: PredictRequest):
    """
    Predict anomalies for a given transaction.
    
    Args:
        request: PredictRequest object with transactions field
    
    Returns:
        dict with prediction results (scores, predictions, is_anomaly)
    """
    try:
        # Extract the transaction data from the request
        data = request.transactions
        
        # Load model and mappings
        model = load_model()
        cat_mappings = load_category_mappings()
        
        # Convert the incoming data to a dictionary
        raw_data = data.dict()
        
        # Preprocess input
        processed_data = preprocess_input(raw_data, cat_mappings)
        
        # Predict anomaly
        scores = model.decision_function(processed_data)
        predictions = (scores < -0.5).astype(int)
        
        return {
            "scores": scores.tolist(),
            "predictions": predictions.tolist(),
            "is_anomaly": bool(np.any(predictions == 1))
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e), "traceback": traceback.format_exc()}

# Add a root endpoint for testing
@app.get("/")
def root():
    return {"message": "API is running"}

# Add a file check endpoint
@app.get("/check-files")
def check_files():
    model_exists = os.path.exists(MODEL_PATH)
    mapping_exists = os.path.exists(MAPPING_PATH)
    return {
        "model_path": MODEL_PATH,
        "model_exists": model_exists,
        "mapping_path": MAPPING_PATH,
        "mapping_exists": mapping_exists
    }