import numpy as np
import logging
import json
from pathlib import Path
import joblib
import pandas as pd
import os
import traceback

# --- Configuration ---
# IMPORTANT: Update these paths if your model/mapping files are located elsewhere!
MODEL_PATH = "/app/models/isolation_forest_model.pkl"
MAPPING_PATH = "/app/models/category_mappings.json"

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Core Functions (Adapted from previous example) ---

def load_model(model_path):
    """Load the trained Isolation Forest model."""
    logger.info(f"Attempting to load model from path: '{model_path}'")
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at specified path: '{model_path}'")
        raise FileNotFoundError(f"Model not found at {model_path}")

    try:
        logger.info(f"Calling joblib.load('{model_path}')")
        loaded_model = joblib.load(model_path)
        logger.info(f"Model loaded successfully. Object type: {type(loaded_model)}")
        if loaded_model is None:
            logger.error("CRITICAL: joblib.load returned None unexpectedly!")
            raise ValueError("Model loading resulted in None unexpectedly.")
        return loaded_model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}", exc_info=True)
        raise


def load_category_mappings(mapping_path):
    """Load categorical encoding mappings from JSON."""
    logger.info(f"Loading category mappings from {mapping_path}")
    if not os.path.exists(mapping_path):
        logger.error(f"Mapping file not found at specified path: '{mapping_path}'")
        raise FileNotFoundError(f"Mapping file not found at {mapping_path}")

    try:
        with open(mapping_path, 'r') as f:
            mappings = json.load(f)
        logger.info("Category mappings loaded successfully.")
        return mappings
    except Exception as e:
        logger.error(f"Failed to load category mappings: {e}", exc_info=True)
        raise


def preprocess_input(data: dict, cat_mappings: dict):
    """Preprocess the input data dictionary and encode categorical columns."""
    logger.info("Starting preprocessing...")
    try:
        # Create DataFrame from the single transaction dictionary
        df = pd.DataFrame([data])
        logger.debug(f"Initial DataFrame columns: {df.columns.tolist()}")

        # Apply category mappings
        for col, mapping in cat_mappings.items():
            if col in df.columns:
                original_value = df[col].iloc[0] # Get value before mapping
                mapped_value = df[col].map(mapping).fillna(-1).astype(int)
                df[col] = mapped_value
                logger.debug(f"Mapped column '{col}': '{original_value}' -> {mapped_value.iloc[0]}")
            else:
                # If a mapped column is entirely missing from input, add it with default value
                logger.warning(f"Column '{col}' expected by mappings not found in input data. Filling with -1.")
                df[col] = -1

        logger.info(f"Preprocessing complete. Final columns: {df.columns.tolist()}")
        logger.debug(f"Preprocessed data head:\n{df.head()}")
        return df

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        raise


def run_prediction(transaction_data: dict):
    
    logger.info("--- Starting Prediction Run ---")
    try:
        # 1. Load model and mappings
        model = load_model(MODEL_PATH)
        cat_mappings = load_category_mappings(MAPPING_PATH)

        # 2. Preprocess the input dictionary
        processed_data = preprocess_input(transaction_data, cat_mappings)

        # 3. Align columns with model features (optional but recommended)
        if hasattr(model, 'feature_names_in_'):
            model_features = model.feature_names_in_
            current_features = processed_data.columns.tolist()
            missing_features = set(model_features) - set(current_features)
            
            if missing_features:
                logger.warning(f"Input data is missing features expected by the model: {missing_features}. Adding with default value 0.")
                for feature in missing_features:
                    processed_data[feature] = 0 # Use an appropriate default

            # Add potentially missing columns AND ensure correct order
            processed_data = processed_data.reindex(columns=model_features, fill_value=0)
            logger.info("Data columns aligned and ordered according to model features.")
        else:
            logger.warning("Model object does not have 'feature_names_in_'. Skipping explicit column alignment/ordering. Ensure preprocessed data matches training features.")


        # 4. Predict anomaly score
        logger.info("Making prediction with the model...")
        # Ensure data is in the format expected by decision_function (usually 2D)
        if isinstance(processed_data, pd.Series):
            predict_input = processed_data.values.reshape(1, -1)
        elif isinstance(processed_data, pd.DataFrame):
            predict_input = processed_data
        else:
            # Convert other types if necessary, e.g., numpy array
            predict_input = np.array(processed_data).reshape(1,-1) # Fallback assumption

        scores = model.decision_function(predict_input) # Should return array of scores

        # 5. Determine prediction based on threshold
        # Lower scores are more anomalous for Isolation Forest
        threshold = -0.5
        prediction = 1 if scores[0] < threshold else 0 # 1 if anomaly, 0 otherwise
        is_anomaly = (prediction == 1)

        logger.info(f"Prediction complete. Score: {scores[0]:.4f}, Prediction: {prediction} (Anomaly={is_anomaly})")

        # 6. Format and return results for the single transaction
        return {
            "score": float(scores[0]),  # Single score
            "prediction": int(prediction), # Single prediction
            "is_anomaly": bool(is_anomaly)
        }

    except FileNotFoundError as e:
        logger.error(f"Prediction failed: Critical file not found - {e}")
        return None # Indicate failure
    except Exception as e:
        logger.error(f"An error occurred during the prediction process: {e}", exc_info=True)
        return None # Indicate failure


# --- Main Execution Block ---
if __name__ == "__main__":
    # Sample input data (as provided in the prompt)
    sample_transaction = {
        "cc_num": "2291163933867244",
        "merchant": "fraud_Kirlin and Sons",
        "category": "personal_care",
        "amt": 2.86,
        "first": "Jeff",
        "last": "Elliott",
        "gender": "M",
        "street": "351 Darlene Green",
        "city": "Anytown",
        "state": "CA",
        "zip": "12345",
        "lat": 37.7749,
        "long": -80.9355,
        "city_pop": 333497,
        "job": "Mechanical engineer",
        "dob": "1968-03-19",
        "unix_time": 1371816865,
        "merch_lat": 33.986391,
        "merch_long": -81.200714
    }

    logger.info("Running standalone prediction script...")
    print("-" * 30)
    print("Input Transaction:")
    # Pretty print the input dictionary
    print(json.dumps(sample_transaction, indent=2))
    print("-" * 30)

    # Run the prediction
    result = run_prediction(sample_transaction)

    print("-" * 30)
    if result:
        print("Prediction Result:")
        # Pretty print the result dictionary
        print(json.dumps(result, indent=2))
    else:
        print("Prediction failed. Check logs above for errors.")
    print("-" * 30)