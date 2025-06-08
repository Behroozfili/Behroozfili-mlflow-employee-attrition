# src/features/build_features.py

import pandas as pd
import numpy as np
import os
import argparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib # For saving the preprocessor

# Assuming logger.py is in src/utils/
try:
    from ..utils.logger import get_logger
except ImportError: # For direct script execution or testing
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.logger import get_logger


logger = get_logger(__name__)

def identify_feature_types(df: pd.DataFrame, target_column: str = 'Attrition'):
    """Identifies numerical and categorical columns, excluding the target and known identifiers/constants."""
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Exclude target column if present
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)

    # Exclude known identifiers or constant columns that won't be used as features
    cols_to_always_exclude = ['EmployeeNumber', 'EmployeeCount', 'StandardHours']
    # 'Over18' is often categorical and constant ('Y'), if numeric and constant it will be caught by nunique check.
    # We will also check for other constant columns dynamically.

    final_numerical_cols = []
    for col in numerical_cols:
        if col not in cols_to_always_exclude:
            if df[col].nunique(dropna=False) > 1: # Only include if not constant
                final_numerical_cols.append(col)
            else:
                logger.info(f"Excluding constant numerical column: {col}")
        else:
            logger.info(f"Excluding pre-defined non-feature numerical column: {col}")


    final_categorical_cols = []
    for col in categorical_cols:
        if col not in cols_to_always_exclude: # 'Over18' might be in this list
            if df[col].nunique(dropna=False) > 1: # Only include if not constant
                final_categorical_cols.append(col)
            else:
                logger.info(f"Excluding constant categorical column: {col}")
        else: # Over18 is often a known constant categorical feature
            logger.info(f"Excluding pre-defined non-feature categorical column: {col}")
    
    # Special handling for 'Over18' if it's categorical (it usually is 'Y')
    if 'Over18' in final_categorical_cols and df['Over18'].nunique(dropna=False) == 1:
        logger.info(f"Excluding constant categorical column: Over18")
        final_categorical_cols.remove('Over18')


    logger.info(f"Identified Numerical Features: {final_numerical_cols}")
    logger.info(f"Identified Categorical Features: {final_categorical_cols}")
    
    return final_numerical_cols, final_categorical_cols

def build_preprocessor(numerical_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Creates a scikit-learn ColumnTransformer for preprocessing.
    """
    logger.info("Building preprocessor pipeline...")

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=False for dense array
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='drop' # Drop any columns not specified (e.g., IDs, constants)
    )
    logger.info("Preprocessor pipeline built successfully.")
    return preprocessor

def build_features(data_path: str, output_path: str, preprocessor_save_path: str):
    """
    Loads split data, builds features using a preprocessor,
    saves the processed data and the preprocessor.
    """
    logger.info(f"Starting feature building. Data source: {data_path}, Output to: {output_path}")

    # Load data
    try:
        X_train_path = os.path.join(data_path, "X_train.csv")
        X_test_path = os.path.join(data_path, "X_test.csv")
        # y_train and y_test are not transformed by this script but need to be passed through
        y_train_path = os.path.join(data_path, "y_train.csv")
        y_test_path = os.path.join(data_path, "y_test.csv")

        X_train_raw = pd.read_csv(X_train_path)
        X_test_raw = pd.read_csv(X_test_path)
        y_train = pd.read_csv(y_train_path).squeeze() # Squeeze to make it a Series
        y_test = pd.read_csv(y_test_path).squeeze()   # Squeeze to make it a Series
        logger.info("Successfully loaded X_train, X_test, y_train, y_test.")
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

    # Identify feature types using the training data
    # (to avoid data leakage from test set during identification)
    numerical_features, categorical_features = identify_feature_types(X_train_raw)

    if not numerical_features and not categorical_features:
        logger.error("No features identified for processing. Check data or identify_feature_types function.")
        raise ValueError("No features to process.")

    # Build the preprocessor based on identified features
    preprocessor = build_preprocessor(numerical_features, categorical_features)

    # Fit preprocessor on training data and transform
    logger.info("Fitting preprocessor on X_train_raw and transforming...")
    X_train_processed_array = preprocessor.fit_transform(X_train_raw)
    logger.info(f"X_train_raw transformed. Shape: {X_train_processed_array.shape}")

    # Transform test data
    logger.info("Transforming X_test_raw...")
    X_test_processed_array = preprocessor.transform(X_test_raw)
    logger.info(f"X_test_raw transformed. Shape: {X_test_processed_array.shape}")

    # Get feature names after one-hot encoding for creating DataFrames
    try:
        # For scikit-learn >= 0.23 (ColumnTransformer.get_feature_names_out)
        # For older versions, onehot.get_feature_names(categorical_features) was used inside.
        # Here we rely on ColumnTransformer's method directly.
        feature_names_out = preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback for older scikit-learn versions (might need more complex logic for feature names)
        logger.warning("preprocessor.get_feature_names_out() not available. Using generic feature names.")
        num_output_features = X_train_processed_array.shape[1]
        feature_names_out = [f"feature_{i}" for i in range(num_output_features)]

    # Convert processed arrays back to DataFrames
    X_train_processed = pd.DataFrame(X_train_processed_array, columns=feature_names_out, index=X_train_raw.index)
    X_test_processed = pd.DataFrame(X_test_processed_array, columns=feature_names_out, index=X_test_raw.index)
    
    logger.info(f"X_train_processed DataFrame shape: {X_train_processed.shape}")
    logger.info(f"X_test_processed DataFrame shape: {X_test_processed.shape}")

    # Create output and preprocessor directories if they don't exist
    os.makedirs(output_path, exist_ok=True)
    preprocessor_dir = os.path.dirname(preprocessor_save_path)
    if preprocessor_dir: # Ensure preprocessor_dir is not empty (e.g. if path is just "preprocessor.joblib")
        os.makedirs(preprocessor_dir, exist_ok=True)
    
    # Save processed data
    processed_X_train_path = os.path.join(output_path, "X_train_processed.csv")
    processed_X_test_path = os.path.join(output_path, "X_test_processed.csv")
    processed_y_train_path = os.path.join(output_path, "y_train_processed.csv") # y doesn't change here, but good to save with new naming
    processed_y_test_path = os.path.join(output_path, "y_test_processed.csv")

    try:
        X_train_processed.to_csv(processed_X_train_path, index=False)
        X_test_processed.to_csv(processed_X_test_path, index=False)
        y_train.to_csv(processed_y_train_path, index=False, header=True)
        y_test.to_csv(processed_y_test_path, index=False, header=True)
        logger.info(f"Successfully saved processed features to {output_path}")
    except Exception as e:
        logger.error(f"Error saving processed feature data: {e}")
        raise

    # Save the preprocessor
    try:
        joblib.dump(preprocessor, preprocessor_save_path)
        logger.info(f"Preprocessor (ColumnTransformer) saved to {preprocessor_save_path}")
    except Exception as e:
        logger.error(f"Error saving preprocessor: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build features for the model.")
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="data/processed", 
        help="Path to directory containing X_train.csv, X_test.csv, etc."
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="data/processed", 
        help="Path to save processed feature files (X_train_processed.csv, etc.). Should be same as data_path if overwriting."
    )
    parser.add_argument(
        "--preprocessor_save_path", 
        type=str, 
        default="models/preprocessor.joblib", 
        help="Path to save the fitted preprocessor object."
    )
    
    args = parser.parse_args()
    
    build_features(args.data_path, args.output_path, args.preprocessor_save_path)
    logger.info("Feature building script finished.")