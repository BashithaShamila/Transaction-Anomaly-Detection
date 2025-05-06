import numpy as np
import logging
import json
from pathlib import Path # Path is imported but not extensively used in this version
import joblib
import pandas as pd
import os
import traceback
import tensorflow as tf # Added for Keras model loading

# --- Configuration ---
# IMPORTANT: Update these paths if your model/mapping files are located elsewhere!
MODEL_PATH = "/app/models/isolation_forest_model.pkl"
MAPPING_PATH = "/app/models/category_mappings.json"

# --- NEW: Configuration for Autoencoder Model ---
AUTOENCODER_MODEL_PATH = "/app/models/autoencoder_model_scaled.keras"
SCALER_PATH = "/app/models/scale.pkl" # Path to the scaler used with the autoencoder

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Core Functions (Adapted from previous example - NO CHANGES TO EXISTING FUNCTIONS BELOW) ---

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
    """Runs prediction using the Isolation Forest model."""
    logger.info("--- Starting Prediction Run (Isolation Forest) ---")
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
            predict_input = processed_data.values # Pass numpy array
        else:
            predict_input = np.array(processed_data).reshape(1,-1)

        scores = model.decision_function(predict_input) 

        threshold = -0.5 # Lower scores are more anomalous
        prediction = 1 if scores[0] < threshold else 0 
        is_anomaly = (prediction == 1)

        logger.info(f"Prediction complete. Score: {scores[0]:.4f}, Prediction: {prediction} (Anomaly={is_anomaly})")

        return {
            "score": float(scores[0]),
            "prediction": int(prediction),
            "is_anomaly": bool(is_anomaly)
        }

    except FileNotFoundError as e:
        logger.error(f"Prediction failed: Critical file not found - {e}")
        return None
    except Exception as e:
        logger.error(f"An error occurred during the prediction process: {e}", exc_info=True)
        return None

# --- END OF UNCHANGED EXISTING FUNCTIONS ---


# --- NEW FUNCTIONS FOR AUTOENCODER MODEL ---

def load_keras_model(model_path: str):
    """Load the trained Keras autoencoder model."""
    logger.info(f"Attempting to load Keras model from path: '{model_path}'")
    if not os.path.exists(model_path):
        logger.error(f"Keras model file not found at specified path: '{model_path}'")
        raise FileNotFoundError(f"Keras model not found at {model_path}")
    try:
        # Suppress TensorFlow INFO/WARNING messages, show only errors
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
        tf.get_logger().setLevel('ERROR')

        loaded_model = tf.keras.models.load_model(model_path)
        logger.info(f"Keras model loaded successfully from '{model_path}'.")
        if loaded_model is None:
            logger.error("CRITICAL: tf.keras.models.load_model returned None unexpectedly!")
            raise ValueError("Keras model loading resulted in None unexpectedly.")
        return loaded_model
    except Exception as e:
        logger.error(f"Failed to load Keras model from {model_path}: {e}", exc_info=True)
        raise

def load_scaler(scaler_path: str):
    """Load the trained scaler object (e.g., StandardScaler)."""
    logger.info(f"Attempting to load scaler from path: '{scaler_path}'")
    if not os.path.exists(scaler_path):
        logger.error(f"Scaler file not found at specified path: '{scaler_path}'")
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    try:
        scaler = joblib.load(scaler_path)
        logger.info(f"Scaler loaded successfully from '{scaler_path}'. Type: {type(scaler)}")
        if scaler is None:
            logger.error("CRITICAL: joblib.load for scaler returned None unexpectedly!")
            raise ValueError("Scaler loading resulted in None unexpectedly.")
        return scaler
    except Exception as e:
        logger.error(f"Failed to load scaler from {scaler_path}: {e}", exc_info=True)
        raise

def preprocess_input_for_autoencoder(data: dict, cat_mappings: dict, scaler, model_feature_names: list):
    """
    Preprocess the input data dictionary for the autoencoder:
    1. Create initial DataFrame from input data.
    2. Apply categorical encoding based on cat_mappings.
    3. Construct a feature DataFrame based on model_feature_names,
       populating from input/mapped data or using defaults for missing features.
    4. Apply scaling to this feature DataFrame.
    """
    logger.info("Starting preprocessing for autoencoder...")
    try:
        # Step 1: Create initial DataFrame from input data
        input_df = pd.DataFrame([data])
        logger.debug(f"Initial input DataFrame for autoencoder columns: {input_df.columns.tolist()}")

        # Step 2: Apply categorical encoding
        temp_processed_df = pd.DataFrame(index=input_df.index)

        for mapped_col_name, mapping_dict in cat_mappings.items():
            if mapped_col_name in input_df.columns:
                original_value = input_df[mapped_col_name].iloc[0]
                temp_processed_df[mapped_col_name] = input_df[mapped_col_name].map(mapping_dict).fillna(-1).astype(int)
                logger.debug(f"Autoencoder: Mapped column '{mapped_col_name}': '{original_value}' -> {temp_processed_df[mapped_col_name].iloc[0]}")
        
        for col in input_df.columns:
            if col not in cat_mappings: # Carry over non-mapped (assumed numerical) columns
                temp_processed_df[col] = input_df[col]
        
        logger.debug(f"Autoencoder: DataFrame after categorical mapping & carry-over: Columns {temp_processed_df.columns.tolist()}")

        # Step 3: Construct the final feature DataFrame based on model_feature_names
        df_for_scaling = pd.DataFrame(columns=model_feature_names, index=input_df.index)

        for feature_name in model_feature_names:
            if feature_name in temp_processed_df.columns:
                df_for_scaling[feature_name] = temp_processed_df[feature_name]
            else:
                if feature_name in cat_mappings:
                    logger.warning(f"Autoencoder: Categorical feature '{feature_name}' (expected by model) "
                                   f"was not in input data. Filling with default mapped value -1.")
                    df_for_scaling[feature_name] = -1
                else: 
                    logger.warning(f"Autoencoder: Numerical feature '{feature_name}' (expected by model) "
                                   f"was not in input data. Filling with 0.0.")
                    df_for_scaling[feature_name] = 0.0
        
        try:
            df_for_scaling = df_for_scaling.astype(float)
        except ValueError as ve:
            logger.error(f"Autoencoder: Failed to convert df_for_scaling to float. Columns: {df_for_scaling.columns}. Dtypes: {df_for_scaling.dtypes}. Error: {ve}")
            # Log problematic columns/values if possible
            for col in df_for_scaling.columns:
                try:
                    df_for_scaling[col].astype(float)
                except ValueError:
                    logger.error(f"Problematic column for float conversion: {col}, unique values: {df_for_scaling[col].unique()[:5]}")
            raise

        logger.debug(f"Autoencoder: DataFrame prepared for scaling (cols: {model_feature_names}):\n{df_for_scaling.head()}")

        # Step 4: Apply scaling
        # ** NEW SCALING STEP IS HERE **
        if hasattr(scaler, 'feature_names_in_'):
            scaler_features_on_load = list(scaler.feature_names_in_)
            if scaler_features_on_load != model_feature_names:
                logger.warning(f"Autoencoder: Scaler feature names/order {scaler_features_on_load} "
                               f"differs from target model feature names/order {model_feature_names}. "
                               "Attempting to use scaler's feature order for transform, then reorder to model's.")
                if set(scaler_features_on_load) != set(model_feature_names):
                    raise ValueError(f"Critical: Feature sets for scaler {set(scaler_features_on_load)} and model {set(model_feature_names)} differ. Cannot proceed.")
                df_to_transform = df_for_scaling[scaler_features_on_load] # Use scaler's feature order
                scaled_values = scaler.transform(df_to_transform)
                df_scaled_temp = pd.DataFrame(scaled_values, columns=scaler_features_on_load, index=df_for_scaling.index)
                df_scaled = df_scaled_temp[model_feature_names] # Reorder to model's expected feature order
            else: 
                scaled_values = scaler.transform(df_for_scaling)
                df_scaled = pd.DataFrame(scaled_values, columns=model_feature_names, index=df_for_scaling.index)
        elif hasattr(scaler, 'n_features_in_'):
            logger.warning("Autoencoder: Scaler does not have 'feature_names_in_'. Assuming it was trained on features "
                           f"in the same quantity ({scaler.n_features_in_}) and order as 'model_feature_names' ({len(model_feature_names)}).")
            if scaler.n_features_in_ != len(model_feature_names):
                raise ValueError(f"Scaler expects {scaler.n_features_in_} features, but model_feature_names has {len(model_feature_names)} features.")
            scaled_values = scaler.transform(df_for_scaling)
            df_scaled = pd.DataFrame(scaled_values, columns=model_feature_names, index=df_for_scaling.index)
        else:
            raise AttributeError("Scaler object does not have 'feature_names_in_' or 'n_features_in_'. Cannot verify feature compatibility.")

        logger.debug(f"Autoencoder: Data after scaling:\n{df_scaled.head()}")
        logger.info(f"Preprocessing for autoencoder complete. Final columns for model: {df_scaled.columns.tolist()}")
        return df_scaled

    except Exception as e:
        logger.error(f"Preprocessing for autoencoder failed: {e}", exc_info=True)
        raise


def run_autoencoder_prediction(transaction_data: dict):
    """Runs prediction using the Keras Autoencoder model."""
    logger.info("--- Starting Prediction Run (Autoencoder) ---")
    try:
        # 1. Load model, mappings, and scaler
        autoencoder_model = load_keras_model(AUTOENCODER_MODEL_PATH)
        cat_mappings = load_category_mappings(MAPPING_PATH)
        scaler = load_scaler(SCALER_PATH)

        # Determine expected features for the autoencoder model
        expected_model_features = None
        if hasattr(scaler, 'feature_names_in_'):
            expected_model_features = list(scaler.feature_names_in_)
            logger.info(f"Autoencoder: Model expected features derived from scaler.feature_names_in_: {expected_model_features}")
        elif hasattr(scaler, 'n_features_in_') and hasattr(autoencoder_model, 'input_shape'):
            # This is a less ideal fallback: requires manual feature list definition if names are important beyond order
            num_scaler_features = scaler.n_features_in_
            # Keras input_shape for Dense layers is typically (None, num_features)
            num_model_features = autoencoder_model.input_shape[-1]
            if num_scaler_features != num_model_features:
                 raise ValueError(f"Mismatch: Scaler expects {num_scaler_features} features, Keras model expects {num_model_features} features.")
            logger.warning(f"Autoencoder: Scaler lacks 'feature_names_in_'. Relying on feature count ({num_scaler_features}). "
                           "A predefined, ordered list of feature names would be more robust if column names matter beyond count/order for preprocessing.")
            # IF YOU REACH HERE, you might need to define `expected_model_features` manually if names are crucial for `preprocess_input_for_autoencoder` beyond just count.
            # For now, `preprocess_input_for_autoencoder` needs names if cat_mappings are involved by name.
            # This path is problematic if `cat_mappings` keys are essential and not just an ordered list.
            raise ValueError("Critical: Scaler does not provide 'feature_names_in_'. "
                             "The current `preprocess_input_for_autoencoder` relies on named features. "
                             "Ensure your scaler is saved with feature names (e.g., fit on Pandas DataFrame) "
                             "or modify preprocessing to work with ordered features if names are not available.")

        else:
            raise ValueError("Critical: Cannot determine expected features for the autoencoder model (scaler lacks 'feature_names_in_' and 'n_features_in_').")

        # 2. Preprocess the input dictionary (including scaling)
        processed_data_df = preprocess_input_for_autoencoder(
            transaction_data, 
            cat_mappings, 
            scaler, 
            model_feature_names=expected_model_features
        )

        # 3. Predict with Autoencoder (get reconstructions)
        logger.info("Making prediction with the autoencoder model...")
        input_for_keras_model = processed_data_df.values # Keras expects NumPy array
        reconstructions = autoencoder_model.predict(input_for_keras_model, verbose=0) # verbose=0 to reduce Keras console output

        # 4. Calculate Reconstruction Error (Mean Squared Error)
        mse = np.mean(np.power(input_for_keras_model - reconstructions, 2), axis=1)
        reconstruction_error = mse[0] 

        # 5. Determine anomaly based on threshold
        autoencoder_threshold = 0.1  # !!! EXAMPLE THRESHOLD - ADJUST BASED ON YOUR MODEL AND DATA !!!
        is_anomaly = reconstruction_error > autoencoder_threshold
        prediction = 1 if is_anomaly else 0

        logger.info(f"Autoencoder prediction complete. Reconstruction Error: {reconstruction_error:.6f}, Prediction: {prediction} (Anomaly={is_anomaly})")

        return {
            "reconstruction_error": float(reconstruction_error),
            "prediction": int(prediction),
            "is_anomaly": bool(is_anomaly)
        }

    except FileNotFoundError as e:
        logger.error(f"Autoencoder prediction failed: Critical file not found - {e}")
        return None
    except Exception as e:
        logger.error(f"An error occurred during the autoencoder prediction process: {e}", exc_info=True)
        return None

# --- Main Execution Block (Updated) ---
if __name__ == "__main__":
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
        "state": "CA", # This will be mapped if 'state' is in category_mappings.json
        "zip": "12345",
        "lat": 37.7749,
        "long": -80.9355,
        "city_pop": 333497,
        "job": "Mechanical engineer",
        "dob": "1968-03-19",
        "unix_time": 1371816865,
        "merch_lat": 33.986391,
        "merch_long": -81.200714
        # Ensure all features expected by your models (after mapping) are present here or handled by preprocessing
    }

    logger.info("Running standalone prediction script...")
    print("-" * 30)
    print("Input Transaction:")
    print(json.dumps(sample_transaction, indent=2))
    print("-" * 30)

    # --- Run Isolation Forest Prediction ---
    logger.info("--- Evaluating Isolation Forest Model ---")
    result_isoforest = run_prediction(sample_transaction)
    print("-" * 30)
    if result_isoforest:
        print("Isolation Forest Prediction Result:")
        print(json.dumps(result_isoforest, indent=2))
    else:
        print("Isolation Forest prediction failed. Check logs above for errors.")
    print("-" * 30)

    # --- Run Autoencoder Prediction ---
    logger.info("\n--- Evaluating Autoencoder Model ---")
    print("\n" + "=" * 30)
    # The input transaction is the same
    result_autoencoder = run_autoencoder_prediction(sample_transaction)
    print("-" * 30)
    if result_autoencoder:
        print("Autoencoder Prediction Result:")
        print(json.dumps(result_autoencoder, indent=2))
    else:
        print("Autoencoder prediction failed. Check logs above for errors.")
    print("=" * 30)