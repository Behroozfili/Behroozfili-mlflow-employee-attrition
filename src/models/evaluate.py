import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
import os 

# Assuming logger.py is in src/utils/
try:
    from ..utils.logger import get_logger
except ImportError: # For direct script execution or testing
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..')) # Go up two levels for src
    from src.utils.logger import get_logger

logger = get_logger(__name__)

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, log_to_mlflow: bool = True):
    """
    Evaluates the model and logs metrics and artifacts to MLflow.
    """
    logger.info("Evaluating model...")
    
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilities for the positive class (class 1)
    except AttributeError: # For models that don't have predict_proba (e.g., some regressors if misused)
        logger.warning(f"Model {model.__class__.__name__} does not have predict_proba method. ROC AUC and LogLoss cannot be calculated.")
        y_pred_proba = None
    except Exception as e:
        logger.error(f"Error during model prediction: {e}")
        if log_to_mlflow and mlflow.active_run():
            mlflow.log_metric("evaluation_status", 0) # 0 for failure
        raise

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    if y_pred_proba is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            logloss = log_loss(y_test, model.predict_proba(X_test)) # Use full probabilities for log_loss
            metrics_dict["roc_auc"] = roc_auc
            metrics_dict["log_loss"] = logloss
            logger.info(f"  ROC AUC: {roc_auc:.4f}")
            logger.info(f"  Log Loss: {logloss:.4f}")
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC or LogLoss: {e}")
            metrics_dict["roc_auc"] = np.nan # Or 0, or skip
            metrics_dict["log_loss"] = np.nan
    else:
        metrics_dict["roc_auc"] = np.nan
        metrics_dict["log_loss"] = np.nan


    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")

    if log_to_mlflow and mlflow.active_run():
        active_run = mlflow.active_run()
        if active_run:
            logger.info("Logging metrics to MLflow...")
            for metric_name, metric_value in metrics_dict.items():
                if not pd.isna(metric_value): # Only log if not NaN
                    mlflow.log_metric(metric_name, metric_value)
            mlflow.log_metric("evaluation_status", 1) # 1 for success

            # Log confusion matrix as an artifact (image)
            try:
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=model.classes_ if hasattr(model, 'classes_') else ['0','1'], 
                            yticklabels=model.classes_ if hasattr(model, 'classes_') else ['0','1'])
                ax.set_title('Confusion Matrix')
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                
                mlflow.log_figure(fig, "confusion_matrix.png")
                plt.close(fig) 
                logger.info("Confusion matrix logged to MLflow.")
            except Exception as e:
                logger.error(f"Failed to log confusion matrix: {e}")
        else:
            logger.warning("No active MLflow run to log metrics to.")
    
    return metrics_dict

if __name__ == "__main__":
    logger.warning("evaluate.py is typically not run standalone. It's used by train_model.py.")
    # Dummy example if needed for testing:
    # from sklearn.linear_model import LogisticRegression
    # X_test_dummy = pd.DataFrame(np.random.rand(20, 3), columns=['a', 'b', 'c'])
    # y_test_dummy = pd.Series(np.random.randint(0, 2, 20))
    # model_dummy = LogisticRegression().fit(X_test_dummy, y_test_dummy)
    # print("Running standalone evaluation example (no MLflow logging):")
    # metrics = evaluate_model(model_dummy, X_test_dummy, y_test_dummy, log_to_mlflow=False)
    # print(f"Standalone metrics: {metrics}")