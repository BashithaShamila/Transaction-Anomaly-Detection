import numpy as np
import logging
import json
from pathlib import Path 
import joblib
import pandas as pd
import os
import traceback
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model



# --- Configuration ---
MODEL_PATH = "/app/models/isolation_forest_model.pkl"
MAPPING_PATH = "/app/models/category_mappings.json"
AUTOENCODER_MODEL_PATH = "/app/models/autoencoder_model_scaled.keras"
SCALER_PATH = "/app/models/scaler.pkl" 

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Autoencoder(Model):
    def __init__(self, input_dim, encoding_dim, hidden_dim_1, hidden_dim_2, name="autoencoder", **kwargs):
        super().__init__(name=name, **kwargs) # Pass standard args like name

        # Store dimensions as attributes - needed for get_config AND building layers
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        # Build the encoder layers
        self.encoder = tf.keras.Sequential([
            # Use InputLayer for explicit shape definition
            layers.InputLayer(input_shape=(self.input_dim,), name="enc_input"),
            layers.Dense(self.hidden_dim_2, activation='relu', name="enc_dense_1"),
            layers.BatchNormalization(name="enc_bn_1"),
            layers.Dropout(0.1, name="enc_dropout_1"),
            layers.Dense(self.hidden_dim_1, activation='relu', name="enc_dense_2"),
            layers.BatchNormalization(name="enc_bn_2"),
            layers.Dropout(0.1, name="enc_dropout_2"),
            layers.Dense(self.encoding_dim, activation='relu', name="enc_dense_latent")
        ], name="encoder")

        # Build the decoder layers
        self.decoder = tf.keras.Sequential([
            # Use InputLayer for explicit shape definition
            layers.InputLayer(input_shape=(self.encoding_dim,), name="dec_input"),
            layers.Dense(self.hidden_dim_1, activation='relu', name="dec_dense_1"),
            layers.BatchNormalization(name="dec_bn_1"),
            layers.Dropout(0.1, name="dec_dropout_1"),
            layers.Dense(self.hidden_dim_2, activation='relu', name="dec_dense_2"),
            layers.BatchNormalization(name="dec_bn_2"),
            layers.Dropout(0.1, name="dec_dropout_2"),
            layers.Dense(self.input_dim, activation='linear', name="dec_output")
        ], name="decoder")

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    # get_config should return arguments needed by __init__ plus any base config
    def get_config(self):
        # Get config from the parent class (Model)
        base_config = super().get_config()
        # Add the specific arguments needed to initialize *this* class
        config = {
            "input_dim": self.input_dim,
            "encoding_dim": self.encoding_dim,
            "hidden_dim_1": self.hidden_dim_1,
            "hidden_dim_2": self.hidden_dim_2,
        }
        # Combine base config with custom args
        base_config.update(config)
        return base_config

    # from_config reconstructs the object using the dictionary from get_config
    @classmethod
    def from_config(cls, config):
        # Log the received config for debugging
        logger.debug(f"Autoencoder.from_config received config: {config}")

        # First try to get values from the config
        init_args = {
            "input_dim": config.pop("input_dim", None),
            "encoding_dim": config.pop("encoding_dim", None),
            "hidden_dim_1": config.pop("hidden_dim_1", None),
            "hidden_dim_2": config.pop("hidden_dim_2", None),
            "name": config.pop("name", "autoencoder"),
        }

        # Check if any required args are missing
        required_args = ['input_dim', 'encoding_dim', 'hidden_dim_1', 'hidden_dim_2']
        missing_args = [arg for arg in required_args if init_args[arg] is None]
        
        # If any args are missing, use hardcoded values from training
        if missing_args:
            logger.warning(f"Missing args in config: {missing_args}. Using hardcoded values.")
            
            # Hardcoded values from your training
            default_values = {
                "input_dim": 22,  
                "encoding_dim": 14,
                "hidden_dim_1": 21,
                "hidden_dim_2": 28,
            }
            
            # Fill in missing values with defaults
            for arg in missing_args:
                init_args[arg] = default_values[arg]
                
            logger.info(f"Using dimensions for Autoencoder: {init_args}")

        
        return cls(**init_args, **config)


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
        # --- Check if model has feature_names_in_ attribute ---
        model_features = None
        if hasattr(model, 'feature_names_in_'):
             model_features = model.feature_names_in_
        elif hasattr(model, 'n_features_in_'):           
             logger.warning(f"Model lacks 'feature_names_in_'. Relying on feature order from preprocessing matching training.")
        else:
             logger.warning("Model lacks 'feature_names_in_' and 'n_features_in_'. Cannot guarantee feature alignment.")


        if model_features is not None:
            current_features = processed_data.columns.tolist()
            # Check for missing features compared to the model
            missing_features = set(model_features) - set(current_features)
            if missing_features:
                logger.warning(f"Input data (after preprocessing) is missing features expected by the Isolation Forest model: {missing_features}. Adding with default value 0.")
                for feature in missing_features:
                    processed_data[feature] = 0 # Use an appropriate default

            # Check for extra features compared to the model (and remove them)
            extra_features = set(current_features) - set(model_features)
            if extra_features:
                logger.warning(f"Input data (after preprocessing) has extra features not expected by the Isolation Forest model: {extra_features}. Removing them.")
                processed_data = processed_data.drop(columns=list(extra_features))


            # Ensure columns are in the correct order expected by the model
            processed_data = processed_data[model_features] # Reorder columns to match model
            logger.info("Data columns aligned and ordered according to Isolation Forest model features.")


        # Predict anomaly score
        logger.info("Making prediction with the Isolation Forest model...")
        # Ensure data is in the format expected by decision_function (usually 2D numpy array)
        if isinstance(processed_data, pd.Series):
            predict_input = processed_data.values.reshape(1, -1)
        elif isinstance(processed_data, pd.DataFrame):
            predict_input = processed_data.values # Pass numpy array
        else:
            predict_input = np.array(processed_data).reshape(1,-1)


        # Use decision_function for Isolation Forest anomaly score
        scores = model.decision_function(predict_input)
        # Define threshold (lower scores indicate anomalies in default IF)
        threshold = -0.1 # Example threshold: Adjust based on validation!
        prediction = 1 if scores[0] < threshold else 0
        is_anomaly = (prediction == 1)
        logger.info(f"Isolation Forest Prediction complete. Score: {scores[0]:.4f}, Threshold: {threshold}, Prediction: {prediction} (Anomaly={is_anomaly})")

        return {
            "score": float(scores[0]),
            "prediction": int(prediction),
            "is_anomaly": bool(is_anomaly)
        }

    except FileNotFoundError as e:
        logger.error(f"Isolation Forest Prediction failed: Critical file not found - {e}")
        return None # Return None or raise an exception
    except Exception as e:
        # Catch the UserWarning about feature names and log it differently if desired
        if "X does not have valid feature names" in str(e):
            logger.warning(f"Sklearn UserWarning during prediction: {e}")
            # Continue processing if it's just a warning
        else:
            logger.error(f"An error occurred during the Isolation Forest prediction process: {e}", exc_info=True)
            return None # Return None or raise an exception


def load_keras_model(model_path: str):
    """Load the trained Keras autoencoder model (using custom_objects)."""
    logger.info(f"Attempting to load Keras model from path: '{model_path}' (using custom_objects)")
    if not os.path.exists(model_path):
        logger.error(f"Keras model file not found at specified path: '{model_path}'")
        raise FileNotFoundError(f"Keras model not found at {model_path}")
    try:
        # Suppress TensorFlow INFO/WARNING messages, show only errors
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel('ERROR')

        # --- Use custom_objects to inform Keras about the Autoencoder class ---
        # This is necessary because the decorator method is not supported by the env's Keras version.
        custom_objects = {'Autoencoder': Autoencoder}
        loaded_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        # --- End of change ---

        logger.info(f"Keras model loaded successfully from '{model_path}'. Type: {type(loaded_model)}")
        if loaded_model is None:
            logger.error("CRITICAL: tf.keras.models.load_model returned None unexpectedly!")
            raise ValueError("Keras model loading resulted in None unexpectedly.")
        return loaded_model
    except Exception as e:
        # Log the full traceback for Keras loading errors
        logger.error(f"Failed to load Keras model from {model_path}: {e}\n{traceback.format_exc()}")
        # Provide a more specific hint if it looks like a format incompatibility
        if "Unable to open file" in str(e) or "file format" in str(e).lower():
            logger.error("Hint: This might indicate the .keras file format is incompatible with the installed TensorFlow/Keras version.")
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

        # Step 2: Apply categorical encoding and carry over numericals
        temp_processed_df = pd.DataFrame(index=input_df.index)
        present_columns = set(input_df.columns)

        # Process mapped columns first
        for mapped_col_name, mapping_dict in cat_mappings.items():
            if mapped_col_name in present_columns:
                original_value = input_df[mapped_col_name].iloc[0]
                # Use .get(value, default) for safer mapping if unseen values might occur
                temp_processed_df[mapped_col_name] = input_df[mapped_col_name].map(lambda x: mapping_dict.get(str(x), -1)).fillna(-1).astype(int)
                # temp_processed_df[mapped_col_name] = input_df[mapped_col_name].map(mapping_dict).fillna(-1).astype(int) # Original line
                logger.debug(f"Autoencoder: Mapped column '{mapped_col_name}': '{original_value}' -> {temp_processed_df[mapped_col_name].iloc[0]}")

        # Carry over columns present in input but not in mappings (assume numerical)
        for col in present_columns:
            if col not in cat_mappings:
                try:
                    temp_processed_df[col] = pd.to_numeric(input_df[col])
                except ValueError:
                    logger.warning(f"Column '{col}' not in mappings and couldn't be converted to numeric. Skipping for autoencoder preprocessing.")

        logger.debug(f"Autoencoder: DataFrame after categorical mapping & carry-over: Columns {temp_processed_df.columns.tolist()}")

        # Step 3: Construct the final feature DataFrame based on model_feature_names
        df_for_scaling = pd.DataFrame(columns=model_feature_names, index=input_df.index)
        available_processed_cols = set(temp_processed_df.columns)

        for feature_name in model_feature_names:
            if feature_name in available_processed_cols:
                df_for_scaling[feature_name] = temp_processed_df[feature_name]
            else:
                # Handle features expected by model but not found after initial processing
                # Check if it *should* have been a mapped categorical column
                if feature_name in cat_mappings:
                    logger.warning(f"Autoencoder: Categorical feature '{feature_name}' (expected by model/scaler) "
                                f"was not in input data or could not be mapped. Filling with default mapped value -1.")
                    df_for_scaling[feature_name] = -1
                else:
                    # Assume it's a numerical feature expected by the model/scaler but wasn't in input
                    logger.warning(f"Autoencoder: Numerical feature '{feature_name}' (expected by model/scaler) "
                                f"was not in input data or was non-numeric. Filling with 0.0.")
                    df_for_scaling[feature_name] = 0.0

        # Ensure all columns are numeric before scaling
        try:
            df_for_scaling = df_for_scaling.astype(float)
        except ValueError as ve:
            logger.error(f"Autoencoder: Failed to convert df_for_scaling to float just before scaling. Columns: {df_for_scaling.columns}. Dtypes: {df_for_scaling.dtypes}. Error: {ve}")
            for col in df_for_scaling.columns:
                if not pd.api.types.is_numeric_dtype(df_for_scaling[col]):
                    logger.error(f"Problematic non-numeric column for float conversion: '{col}', example value: {df_for_scaling[col].iloc[0]}, dtype: {df_for_scaling[col].dtype}")
            raise

        logger.debug(f"Autoencoder: DataFrame prepared for scaling (cols ordered by model_feature_names: {model_feature_names}):\n{df_for_scaling.head()}")

        # Step 4: Apply scaling using the provided scaler
        if not hasattr(scaler, 'transform'):
            raise AttributeError("Provided scaler object does not have a 'transform' method.")

        try:
            if hasattr(scaler, 'feature_names_in_'):
                scaler_features = list(scaler.feature_names_in_)
                if set(scaler_features) != set(model_feature_names):
                    raise ValueError(f"Feature mismatch: Scaler expects {set(scaler_features)} but received {set(model_feature_names)} for scaling.")
                df_to_transform = df_for_scaling[scaler_features] # Ensure order matches scaler
                logger.debug(f"Autoencoder: Columns before scaling (ordered as scaler expects): {df_to_transform.columns.tolist()}")
                scaled_values = scaler.transform(df_to_transform)
                df_scaled = pd.DataFrame(scaled_values, columns=scaler_features, index=df_for_scaling.index)
                df_scaled = df_scaled[model_feature_names] # Reorder back to original desired order
            elif hasattr(scaler, 'n_features_in_'):
                logger.warning("Autoencoder: Scaler lacks 'feature_names_in_'. Assuming df_for_scaling has the correct "
                                f"number ({scaler.n_features_in_}) and order of features.")
                if scaler.n_features_in_ != len(model_feature_names):
                    raise ValueError(f"Feature count mismatch: Scaler expects {scaler.n_features_in_}, but received {len(model_feature_names)} features.")
                scaled_values = scaler.transform(df_for_scaling.values) # Pass numpy array if no names
                df_scaled = pd.DataFrame(scaled_values, columns=model_feature_names, index=df_for_scaling.index)
            else:
                logger.warning("Autoencoder: Scaler lacks 'feature_names_in_' and 'n_features_in_'. Attempting scaling assuming correct feature order and number.")
                scaled_values = scaler.transform(df_for_scaling.values) # Pass numpy array
                df_scaled = pd.DataFrame(scaled_values, columns=model_feature_names, index=df_for_scaling.index)

        except Exception as scale_e:
            logger.error(f"Error during scaler transform: {scale_e}", exc_info=True)
            logger.error(f"Data sent to scaler had shape: {df_for_scaling.shape}, columns: {df_for_scaling.columns.tolist()}")
            if hasattr(scaler, 'n_features_in_'):
                logger.error(f"Scaler expected n_features_in_: {scaler.n_features_in_}")
            if hasattr(scaler, 'feature_names_in_'):
                logger.error(f"Scaler expected feature_names_in_: {scaler.feature_names_in_}")
            raise


        logger.debug(f"Autoencoder: Data after scaling (columns ordered by model_feature_names):\n{df_scaled.head()}")
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

        # Determine expected features (rely on scaler's knowledge if available)
        expected_model_features = None
        if hasattr(scaler, 'feature_names_in_'):
            expected_model_features = list(scaler.feature_names_in_)
            logger.info(f"Autoencoder: Model expected features derived from scaler.feature_names_in_: {expected_model_features}")
            # Optional: Verify against loaded Keras model's input dimension
            try:
                 # Try getting input dim from config stored within our custom class instance
                 if isinstance(autoencoder_model, Autoencoder):
                     model_input_dim = autoencoder_model.input_dim
                 else: # Fallback: try layer info (might be less reliable)
                    model_input_dim = autoencoder_model.layers[0].input_shape[-1] # Assumes first layer defines input shape

                 if model_input_dim != len(expected_model_features):
                     logger.warning(f"Potential mismatch: Scaler indicates {len(expected_model_features)} features, "
                                    f"but loaded Keras model seems to expect input dimension {model_input_dim}. Proceeding, but verify model/scaler compatibility.")
            except Exception as shape_e:
                 logger.warning(f"Could not automatically verify Keras model input dimension against scaler features: {shape_e}")

        elif hasattr(scaler, 'n_features_in_'):
             num_scaler_features = scaler.n_features_in_
             logger.warning(f"Autoencoder: Scaler lacks 'feature_names_in_'. Cannot derive feature names automatically. "
                           f"Will attempt to proceed using feature count ({num_scaler_features}) and assuming preprocessing produces the correct order.")
             raise ValueError("Critical: Scaler lacks 'feature_names_in_'. Cannot reliably determine feature names/order for preprocessing. "
                              "Please ensure the scaler is saved with feature names (e.g., fit on a Pandas DataFrame during training).")
        else:
            raise ValueError("Critical: Cannot determine expected features for the autoencoder model (scaler lacks 'feature_names_in_' and 'n_features_in_').")

        # 2. Preprocess the input dictionary (including scaling)
        processed_data_df = preprocess_input_for_autoencoder(
            transaction_data,
            cat_mappings,
            scaler,
            model_feature_names=expected_model_features # Use features derived from scaler
        )

        # 3. Predict with Autoencoder (get reconstructions)
        logger.info("Making prediction with the autoencoder model...")
        input_for_keras_model = processed_data_df.values # Keras expects NumPy array
        reconstructions = autoencoder_model.predict(input_for_keras_model, verbose=0)

        # 4. Calculate Reconstruction Error (Mean Absolute Error - MAE)
        mae = np.mean(np.abs(input_for_keras_model - reconstructions), axis=1)
        reconstruction_error = mae[0] # Get the error for the single prediction

        
        autoencoder_threshold = 99.5  
        is_anomaly = reconstruction_error > autoencoder_threshold
        prediction = 1 if is_anomaly else 0

        logger.info(f"Autoencoder prediction complete. Reconstruction Error (MAE): {reconstruction_error:.6f}, Threshold: {autoencoder_threshold}, Prediction: {prediction} (Anomaly={is_anomaly})")

        return {
            "reconstruction_error": float(reconstruction_error),
            "prediction": int(prediction),
            "is_anomaly": bool(is_anomaly)
        }

    except FileNotFoundError as e:
        logger.error(f"Autoencoder prediction failed: Critical file not found - {e}")
        return None
    except ValueError as e: # Catch specific errors like feature mismatch or missing names
       logger.error(f"Autoencoder prediction failed due to configuration/data issue: {e}", exc_info=True)
       return None
    except Exception as e:
        logger.error(f"An error occurred during the autoencoder prediction process: {e}", exc_info=True)
        return None
    



def fraud_detection_wrapper(transaction):
    logger.info("Starting fraud detection ensemble analysis...")
    
    # Run both models using the existing functions
    result_isoforest = run_prediction(transaction)
    result_autoencoder = run_autoencoder_prediction(transaction)
    
    # Create result object with the original transaction
    result = transaction.copy()
    
    # Get the individual predictions (default to 0 if model failed)
    isoforest_prediction = result_isoforest.get('prediction', 0) if result_isoforest else 0
    autoencoder_prediction = result_autoencoder.get('prediction', 0) if result_autoencoder else 0
    
    # Ensemble logic - consider it fraud if either model flags it
    is_fraud = 1 if (isoforest_prediction == 1 or autoencoder_prediction == 1) else 0
    
    # Add fraud result to the original transaction
    result['is_fraud'] = is_fraud
    
    return result


# --- Main Execution Block ---
if __name__ == "__main__":
    # Sample transaction from your code
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
    
    print(json.dumps(sample_transaction, indent=2))
    
    # Run fraud detection wrapper
    fraud_result = fraud_detection_wrapper(sample_transaction)
    
    # Print the final result
    print("-" * 30)
    print("Fraud Detection Result:")
    print(json.dumps(fraud_result, indent=2))