# src/models/train_model.py

import pandas as pd
import os
import argparse
import mlflow
import mlflow.sklearn # For scikit-learn model logging
from sklearn.linear_model import LogisticRegression # Example model
# from sklearn.ensemble import RandomForestClassifier # Another example
import joblib # For saving the model locally

# Assuming logger.py is in src/utils/ and evaluate.py is in the same directory
try:
    from ..utils.logger import get_logger
    from .evaluate import evaluate_model # Import evaluation function
except ImportError: # For direct script execution or testing
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.logger import get_logger
    # For evaluate, if running directly, we might need to adjust path or it might fail
    # This structure assumes 'python -m src.models.train_model'
    from models.evaluate import evaluate_model


logger = get_logger(__name__)

def train_model(
    processed_data_path: str, # This path now points to where _processed.csv files are
    model_output_path: str,
    experiment_name: str = "EmployeeAttritionExperiment",
    run_name: str = "DefaultRun",
    # preprocessor_path: str = None # No longer needed here as data is already processed
    ):
    """
    Trains a model using preprocessed data, logs parameters and metrics to MLflow,
    and saves the model.
    """
    logger.info(f"Starting model training. Data source: {processed_data_path}")

    # Load PROCESSED training and testing data (output from build_features.py)
    try:
        X_train_path = os.path.join(processed_data_path, "X_train_processed.csv")
        y_train_path = os.path.join(processed_data_path, "y_train_processed.csv")
        X_test_path = os.path.join(processed_data_path, "X_test_processed.csv")
        y_test_path = os.path.join(processed_data_path, "y_test_processed.csv")

        X_train = pd.read_csv(X_train_path)
        # y_train is saved as a DataFrame with a header, so read it and then squeeze
        y_train = pd.read_csv(y_train_path).squeeze() 
        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path).squeeze()
        
        logger.info("Successfully loaded fully processed training and testing data.")
        logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    except FileNotFoundError as e:
        logger.error(f"Fully processed data file not found: {e}")
        logger.error(f"Expected files like X_train_processed.csv in {processed_data_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading fully processed data: {e}")
        raise
    
    # Set MLflow experiment
    try:
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set to: {experiment_name}")
    except Exception as e:
        logger.error(f"Could not set MLflow experiment '{experiment_name}': {e}")
        # Decide if you want to proceed without MLflow or raise error
        # For now, we'll proceed but log a warning
        logger.warning("Proceeding without MLflow experiment tracking due to an error.")


    with mlflow.start_run(run_name=run_name) as run:
        if run:
            run_id = run.info.run_id
            logger.info(f"MLflow Run ID: {run_id}")
            mlflow.log_param("mlflow_run_id", run_id) # Log run_id as a parameter for easy reference
        else:
            logger.warning("Failed to start MLflow run. Metrics and model will not be logged to MLflow.")
            run_id = "local_run_no_mlflow" # Placeholder

        # --- Model Definition and Training ---
        # Example: Logistic Regression
        # You can add more hyperparameters here or make them configurable via argparse
        model_params = {
            "solver": "liblinear",
            "random_state": 42,
            "C": 1.0, # Example hyperparameter
            "penalty": "l1" # Example
        }
        model = LogisticRegression(**model_params)
        
        # Log model parameters to MLflow if run is active
        if run:
            mlflow.log_params(model_params)
            mlflow.log_param("model_type", model.__class__.__name__)
        
        logger.info(f"Training {model.__class__.__name__} with params: {model_params}")
        try:
            model.fit(X_train, y_train)
            logger.info("Model training completed.")
            if run: mlflow.log_metric("training_status", 1) # 1 for success
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            if run: mlflow.log_metric("training_status", 0) # 0 for failure
            # If MLflow run active, set a tag indicating failure
            if run: mlflow.set_tag("pipeline_status", "training_failed")
            raise

        # --- Evaluate Model ---
        logger.info("Evaluating model on the test set...")
        # The evaluate_model function will log its own metrics to MLflow if run is active
        test_metrics = evaluate_model(model, X_test, y_test, log_to_mlflow=(run is not None))
        logger.info(f"Test set metrics: {test_metrics}")

        # --- Log Model to MLflow Registry (if run is active) ---
        if run:
            logger.info("Logging model to MLflow registry...")
            try:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="attrition-model", # Path within MLflow run artifacts
                    registered_model_name="EmployeeAttritionModel" # Name in Model Registry
                    # You can add signature and input_example for better model serving
                    # from mlflow.models.signature import infer_signature
                    # signature = infer_signature(X_train, model.predict(X_train))
                    # input_example = X_train.head(5)
                    # mlflow.sklearn.log_model(..., signature=signature, input_example=input_example)
                )
                logger.info(f"Model logged to MLflow with artifact_path 'attrition-model' and registered_model_name 'EmployeeAttritionModel'")
            except Exception as e:
                logger.error(f"Failed to log model to MLflow: {e}")
        else:
            logger.warning("MLflow run not active, skipping model logging to MLflow.")


        # --- Save Model Locally (Always do this as a backup) ---
        os.makedirs(model_output_path, exist_ok=True)
        # Use a more descriptive name for the locally saved model, perhaps including the run_name or timestamp
        model_filename_base = f"trained_model_{model.__class__.__name__}_{run_id}.joblib"
        model_filename = os.path.join(model_output_path, model_filename_base)
        try:
            joblib.dump(model, model_filename)
            logger.info(f"Model saved locally to {model_filename}")
            if run: # Also log the joblib file as an artifact if MLflow is active
                 mlflow.log_artifact(model_filename, "local_model_backup")
        except Exception as e:
            logger.error(f"Error saving model locally: {e}")
            # Even if local save fails, don't necessarily fail the whole run if MLflow logging worked
            
        if run: mlflow.set_tag("pipeline_status", "success")

    logger.info("Model training and logging script finished.")
    return model_filename if 'model_filename' in locals() else None # Return path to locally saved model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model and log with MLflow.")
    parser.add_argument(
        "--processed_data_path", 
        type=str, 
        default="data/processed", 
        help="Path to directory containing X_train_processed.csv, y_train_processed.csv, etc."
    )
    parser.add_argument(
        "--model_output_path", 
        type=str, 
        default="models", 
        help="Path to save the trained model locally."
    )
    parser.add_argument(
        "--experiment_name", 
        type=str, 
        default="EmployeeAttritionExperiment", 
        help="Name of the MLflow experiment."
    )
    parser.add_argument(
        "--run_name", 
        type=str, 
        default=f"Run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}", 
        help="Name for this MLflow run."
    )

    args = parser.parse_args()

    # The preprocessor_path is no longer directly needed by train_model.py
    # as the data is assumed to be already processed by build_features.py
    train_model(
        args.processed_data_path, 
        args.model_output_path, 
        args.experiment_name, 
        args.run_name
    )