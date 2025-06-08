import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split

# Assuming logger.py is in src/utils/
try:
    from ..utils.logger import get_logger
except ImportError: # For direct script execution or testing
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..')) # Go up two levels for src
    from src.utils.logger import get_logger

logger = get_logger(__name__)

def split_data(processed_file_path: str, output_path: str, target_column: str = "Attrition", test_size: float = 0.2, random_state: int = 42):
    """
    Splits data into training and testing sets.
    """
    logger.info(f"Starting data splitting from {processed_file_path}")

    if not os.path.exists(processed_file_path):
        logger.error(f"Processed data file not found at {processed_file_path}")
        raise FileNotFoundError(f"Processed data file not found at {processed_file_path}")

    try:
        df = pd.read_csv(processed_file_path)
        logger.info(f"Successfully loaded {processed_file_path} with shape {df.shape}")
    except Exception as e:
        logger.error(f"Error loading processed CSV file: {e}")
        raise

    if target_column not in df.columns:
        logger.error(f"Target column '{target_column}' not found in the dataframe.")
        raise ValueError(f"Target column '{target_column}' not found.")

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    logger.info(f"Splitting data with test_size={test_size} and random_state={random_state}")
    # Stratify by y if it's a classification task and y has more than 1 class
    stratify_on_y = y if y.nunique() > 1 else None
    if stratify_on_y is None and y.nunique() <=1:
        logger.warning(f"Target column '{target_column}' has only {y.nunique()} unique value(s). Stratification will not be applied.")


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_on_y
    )

    logger.info(f"Train set shape: X_train {X_train.shape}, y_train {y_train.shape}")
    logger.info(f"Test set shape: X_test {X_test.shape}, y_test {y_test.shape}")

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    try:
        X_train.to_csv(os.path.join(output_path, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(output_path, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(output_path, "y_train.csv"), index=False, header=True) # Save y as Series with header
        y_test.to_csv(os.path.join(output_path, "y_test.csv"), index=False, header=True)  # Save y as Series with header
        logger.info(f"Successfully saved train and test sets to {output_path}")
    except Exception as e:
        logger.error(f"Error saving train/test data: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into train and test sets.")
    parser.add_argument("--processed_file_path", type=str, default="data/processed/processed_employee_attrition.csv", help="Path to the processed data file.")
    parser.add_argument("--output_path", type=str, default="data/processed", help="Path to save train and test CSV files.")
    parser.add_argument("--target_column", type=str, default="Attrition", help="Name of the target column.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset to include in the test split.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility.")
    
    args = parser.parse_args()
    
    split_data(args.processed_file_path, args.output_path, args.target_column, args.test_size, args.random_state)
    logger.info("Data splitting script finished.")