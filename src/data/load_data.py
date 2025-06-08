import pandas as pd
import os
import argparse
from ..utils.logger import get_logger # از .utils چون در همان پکیج src هستیم

# فرض می‌کنیم logger.py در پوشه utils در کنار data قرار دارد
# اگر از بیرون src اجرا می‌کنید (مثلا با python -m src.data.load_data) این import کار می‌کند.

logger = get_logger(__name__)

def load_and_preprocess_data(raw_data_path: str, processed_data_path: str, raw_filename: str = "employee_attrition.csv") -> pd.DataFrame:
    """
    Loads raw data, performs minimal preprocessing, and saves it.
    """
    logger.info(f"Starting data loading and preprocessing from {raw_data_path}")
    
    input_file = os.path.join(raw_data_path, raw_filename)
    if not os.path.exists(input_file):
        logger.error(f"Raw data file not found at {input_file}")
        raise FileNotFoundError(f"Raw data file not found at {input_file}")

    try:
        df = pd.read_csv(input_file)
        logger.info(f"Successfully loaded {input_file} with shape {df.shape}")
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        raise

    # Minimal preprocessing example: Convert 'Attrition' to numerical
    if 'Attrition' in df.columns:
        df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
        logger.info("Converted 'Attrition' column to numerical (1 for Yes, 0 for No).")
    else:
        logger.warning("Target column 'Attrition' not found. Skipping conversion.")
        # Consider raising an error if 'Attrition' is essential for downstream tasks
        # raise ValueError("Target column 'Attrition' not found in the dataset.")

    # Create processed data directory if it doesn't exist
    os.makedirs(processed_data_path, exist_ok=True)
    
    processed_filename = "processed_" + raw_filename
    output_file = os.path.join(processed_data_path, processed_filename)
    
    try:
        df.to_csv(output_file, index=False)
        logger.info(f"Successfully saved processed data to {output_file}")
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        raise
        
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and preprocess data.")
    parser.add_argument("--raw_data_path", type=str, default="data/raw", help="Path to raw data directory.")
    parser.add_argument("--processed_data_path", type=str, default="data/processed", help="Path to save processed data.")
    parser.add_argument("--raw_filename", type=str, default="employee_attrition.csv", help="Name of the raw CSV file.")
    
    args = parser.parse_args()
    
    load_and_preprocess_data(args.raw_data_path, args.processed_data_path, args.raw_filename)
    logger.info("Data loading and preprocessing script finished.")