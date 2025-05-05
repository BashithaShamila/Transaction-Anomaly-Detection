# /Users/bashithashamila/Web Development/transaction-anomaly-detection/src/model/fraud_model.py
import numpy as np
import logging
import json
from pathlib import Path
import joblib
import pandas as pd
import os
import traceback

# Logger setup for this module
logger = logging.getLogger(__name__)

# --- CONFIGURATION WITH EXACT PATHS ---
# These paths are relative to the context where this module is used,
# or absolute paths if running in a specific environment (like a container).
MODEL_PATH = "/app/models/isolation_forest_model.pkl"
MAPPING_PATH = "/app/models/category_mappings.json"

# --- Global variables to hold loaded model and mappings (for efficiency) ---
# Initialize to None. They will be loaded on the first prediction request.
# For production, consider loading these during application startup.
_model = None
_cat_mappings = None


def _load_model():
    """Load the trained Isolation Forest model (internal helper)."""
    global _model
    if _model is not None:
        logger.debug("Model already loaded.")
        return _model

    logger.info(f"Attempting to load model from path: '{MODEL_PATH}'")
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at specified path: '{MODEL_PATH}'")
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    try:
        logger.info(f"Calling joblib.load('{MODEL_PATH}')")
        loaded_model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded successfully. Object type: {type(loaded_model)}")

        if loaded_model is None:
            logger.error("CRITICAL: joblib.load returned None unexpectedly!")
            raise ValueError("Model loading resulted in None unexpectedly.")

        _model = loaded_model # Store in global variable
        return _model
    except Exception as e:
        logger.error(f"Failed to load model from {MODEL_PATH}: {e}", exc_info=True)
        raise # Re-raise the exception


def _load_category_mappings():
    """Load categorical encoding mappings from JSON (internal helper)."""
    global _cat_mappings
    if _cat_mappings is not None:
        logger.debug("Category mappings already loaded.")
        return _cat_mappings

    logger.info(f"Loading category mappings from {MAPPING_PATH}")
    if not os.path.exists(MAPPING_PATH):
        logger.error(f"Mapping file not found at specified path: '{MAPPING_PATH}'")
        raise FileNotFoundError(f"Mapping file not found at {MAPPING_PATH}")

    try:
        with open(MAPPING_PATH, 'r') as f:
            mappings = json.load(f)
        _cat_mappings = mappings # Store in global variable
        return mappings
    except Exception as e:
        logger.error(f"Failed to load category mappings: {e}")
        raise # Re-raise the exception


def _preprocess_input(data: dict, cat_mappings: dict):
    """Preprocess the input data dictionary and encode categorical columns."""
    try:
        # Create DataFrame from the single transaction dictionary
        df = pd.DataFrame([data])

        # Apply category mappings
        for col, mapping in cat_mappings.items():
            if col in df.columns:
                # Map known values, fill unknown/missing with -1
                df[col] = df[col].map(mapping).fillna(-1).astype(int)
            else:
                # If a mapped column is entirely missing from input, add it with default value
                logger.warning(f"Column '{col}' expected by mappings not found in input data. Filling with -1.")
                df[col] = -1

        # Select only the columns expected by the model (based on mappings keys)
        # This assumes the mapping keys represent the feature set the model was trained on.
        # If the model expects features not in mappings, this needs adjustment.
        expected_features = list(cat_mappings.keys()) # Or load feature list separately
        
        # Reorder columns and handle missing columns not in mappings (if any)
        # For simplicity, assuming mapping keys cover all necessary categorical features.
        # Numerical features are passed through. Ensure all required features are present.
        # A more robust approach would be to have an explicit list of all features.
        
        # For now, let's assume the model can handle the DataFrame `df` as is after mapping
        # If specific columns are required: df = df[required_column_order]
        
        logger.info(f"Preprocessed data columns: {df.columns.tolist()}")
        logger.debug(f"Preprocessed data head:\n{df.head()}")
        return df

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        raise # Re-raise the exception


def predict_transaction_anomaly(transaction_data: dict):
    """
    Loads model/mappings, preprocesses data, and predicts anomaly for a single transaction.

    Args:
        transaction_data: A dictionary representing a single transaction.

    Returns:
        A dictionary with prediction results: {"scores": list, "predictions": list, "is_anomaly": bool}

    Raises:
        FileNotFoundError: If model or mapping file is not found.
        Exception: If any other error occurs during loading, preprocessing, or prediction.
    """
    try:
        # Load model and mappings (uses cached versions after first load)
        model = _load_model()
        cat_mappings = _load_category_mappings()

        # Preprocess the input dictionary
        processed_data = _preprocess_input(transaction_data, cat_mappings)

        # Ensure columns match model's expected features if possible
        # (This might require storing feature names during training)
        # Example check (requires model to have 'feature_names_in_' attribute):
        if hasattr(model, 'feature_names_in_'):
            # Get features the model expects
            model_features = model.feature_names_in_
            # Get features we currently have
            current_features = processed_data.columns.tolist()
            
            # Check if all model features are present
            missing_features = set(model_features) - set(current_features)
            if missing_features:
                # Handle missing features (e.g., add them with default values)
                logger.warning(f"Input data is missing features expected by the model: {missing_features}. Adding with default value 0.")
                for feature in missing_features:
                    processed_data[feature] = 0 # Or another appropriate default
            
            # Ensure correct column order
            processed_data = processed_data[model_features]
            logger.info("Data columns aligned with model features.")

        # Predict anomaly score
        # Note: predict expects a 2D array-like structure, DataFrame works.
        logger.info("Making prediction with the model...")
        scores = model.decision_function(processed_data)
        
        # Determine prediction based on threshold (e.g., -0.5)
        # Lower scores are more anomalous for Isolation Forest
        predictions = (scores < -0.5).astype(int) # 1 if anomaly, 0 otherwise

        logger.info(f"Prediction complete. Score: {scores[0]}, Prediction: {predictions[0]}")

        return {
            "scores": scores.tolist(), # Return as list
            "predictions": predictions.tolist(), # Return as list
            "is_anomaly": bool(predictions[0] == 1) # True if the single transaction is anomalous
        }

    except FileNotFoundError as fnf_error:
        logger.error(f"File not found error during prediction: {fnf_error}")
        raise # Re-raise specific error for API layer to potentially handle differently

    except Exception as e:
        logger.error(f"Core prediction logic failed: {e}", exc_info=True)
        # Wrap the original exception or raise a custom one
        raise RuntimeError(f"Prediction processing failed: {e}") from e