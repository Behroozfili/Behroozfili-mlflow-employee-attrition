import subprocess
import os
import logging
from datetime import datetime

# Configure basic logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command_list):
    """Executes a shell command and logs its output."""
    logger.info(f"Executing: {' '.join(command_list)}")
    try:
        process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        for line in process.stdout:
            print(line, end='') # Print subprocess output in real-time
        process.wait() # Wait for the process to complete
        if process.returncode != 0:
            logger.error(f"Command failed with exit code {process.returncode}: {' '.join(command_list)}")
            raise subprocess.CalledProcessError(process.returncode, command_list)
        logger.info(f"Successfully executed: {' '.join(command_list)}")
    except Exception as e:
        logger.error(f"An error occurred while executing {' '.join(command_list)}: {e}")
        raise

def main():
    base_dir = os.getcwd() # Assumes script is run from project root
    data_raw_dir = os.path.join(base_dir, "data", "raw")
    data_processed_dir = os.path.join(base_dir, "data", "processed")
    models_dir = os.path.join(base_dir, "models")

    # Ensure Python can find modules in src
    # This might be needed if not running with `python -m` for the main script itself
    # or if `src` is not an installed package.
    env = os.environ.copy()
    python_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{base_dir}{os.pathsep}{python_path}"


    logger.info("--------------------------------------")
    logger.info("Starting Employee Attrition ML Pipeline (Python orchestrator)")
    logger.info("--------------------------------------")

    # Create necessary directories
    os.makedirs(data_processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # --- STEP 1: Load and Preprocess Data ---
    logger.info("\nSTEP 1: Load and Preprocess Data")
    logger.info("=================================")
    cmd_load = [
        "python", "-m", "src.data.load_data",
        "--raw_data_path", data_raw_dir,
        "--processed_data_path", data_processed_dir,
        "--raw_filename", "employee_attrition.csv"
    ]
    run_command(cmd_load)
    logger.info("Load and Preprocess Data - COMPLETED\n")

    processed_csv_name = "processed_employee_attrition.csv"

    # --- STEP 2: Split Data ---
    logger.info("STEP 2: Split Data")
    logger.info("===================")
    cmd_split = [
        "python", "-m", "src.data.split",
        "--processed_file_path", os.path.join(data_processed_dir, processed_csv_name),
        "--output_path", data_processed_dir,
        "--target_column", "Attrition",
        "--test_size", "0.2",
        "--random_state", "42"
    ]
    run_command(cmd_split)
    logger.info("Split Data - COMPLETED\n")

    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")

    # --- STEP 3: Build Features ---
    logger.info("STEP 3: Build Features")
    logger.info("======================")
    cmd_features = [
        "python", "-m", "src.features.build_features",
        "--data_path", data_processed_dir,
        "--output_path", data_processed_dir,
        "--preprocessor_save_path", preprocessor_path
    ]
    run_command(cmd_features)
    logger.info("Build Features - COMPLETED\n")

    experiment_name = "EmployeeAttritionExperiment_PyScript"
    run_name = f"Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # --- STEP 4: Train and Evaluate Model ---
    logger.info("STEP 4: Train and Evaluate Model")
    logger.info("================================")
    cmd_train = [
        "python", "-m", "src.models.train_model",
        "--processed_data_path", data_processed_dir,
        "--model_output_path", models_dir,
        "--experiment_name", experiment_name,
        "--run_name", run_name
    ]
    run_command(cmd_train)
    logger.info("Train and Evaluate Model - COMPLETED\n")

    logger.info("--------------------------------------")
    logger.info("ML Pipeline Finished Successfully!")
    logger.info("--------------------------------------")
    logger.info("To view results, run: mlflow ui")
    logger.info("Then open http://localhost:5000 in your browser.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        exit(1)