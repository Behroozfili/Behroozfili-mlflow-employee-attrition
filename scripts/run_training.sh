#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define base directory relative to the script's location or assume /app in Docker.
# Since this script is in /app/scripts/ inside the container and WORKDIR is /app,
# using absolute paths from /app is clearest.
BASE_DIR="/app"
DATA_RAW_DIR="${BASE_DIR}/data/raw"
DATA_PROCESSED_DIR="${BASE_DIR}/data/processed"
MODELS_DIR="${BASE_DIR}/models"

# PYTHONPATH should already be set by Dockerfile or docker-compose.yml
# export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}" # Usually not needed here

echo "--------------------------------------------------------------------"
echo "Starting Employee Attrition ML Pipeline (via run_training.sh in Docker)"
echo "--------------------------------------------------------------------"

# Directories like data/processed and models should be created by volume mounts
# from the host or exist due to COPY in Dockerfile.
# No need to mkdir them here if volumes are correctly mounted from host.

# --- STEP 1: Load and Preprocess Data ---
echo ""
echo "STEP 1: Load and Preprocess Data"
echo "================================="
python -m src.data.load_data \
    --raw_data_path "${DATA_RAW_DIR}" \
    --processed_data_path "${DATA_PROCESSED_DIR}" \
    --raw_filename "employee_attrition.csv"

if [ $? -ne 0 ]; then
    echo "Error in Step 1: Load and Preprocess Data. Exiting."
    exit 1
fi
echo "Load and Preprocess Data - COMPLETED"
echo ""

# Define the name of the output file from load_data.py
PROCESSED_RAW_CSV_NAME="processed_employee_attrition.csv"

# --- STEP 2: Split Data ---
echo "STEP 2: Split Data"
echo "==================="
python -m src.data.split \
    --processed_file_path "${DATA_PROCESSED_DIR}/${PROCESSED_RAW_CSV_NAME}" \
    --output_path "${DATA_PROCESSED_DIR}" \
    --target_column "Attrition" \
    --test_size 0.2 \
    --random_state 42

if [ $? -ne 0 ]; then
    echo "Error in Step 2: Split Data. Exiting."
    exit 1
fi
echo "Split Data - COMPLETED"
echo ""

# Define the path for the preprocessor object
PREPROCESSOR_SAVE_PATH="${MODELS_DIR}/preprocessor.joblib"

# --- STEP 3: Build Features ---
echo "STEP 3: Build Features"
echo "======================"
python -m src.features.build_features \
    --data_path "${DATA_PROCESSED_DIR}" \
    --output_path "${DATA_PROCESSED_DIR}" \
    --preprocessor_save_path "${PREPROCESSOR_SAVE_PATH}"

if [ $? -ne 0 ]; then
    echo "Error in Step 3: Build Features. Exiting."
    exit 1
fi
echo "Build Features - COMPLETED"
echo ""

# --- STEP 4: Train and Evaluate Model ---
# Generate a unique run name for MLflow using current timestamp
# This ensures each execution of the pipeline is a new run in MLflow.
EXPERIMENT_NAME="EmployeeAttrition_DockerShell" # A descriptive experiment name
RUN_NAME="Run_$(date +%Y%m%d_%H%M%S)" # Example: Run_20230115_143055

echo "STEP 4: Train and Evaluate Model"
echo "Experiment Name: ${EXPERIMENT_NAME}"
echo "Run Name: ${RUN_NAME}"
echo "================================"
python -m src.models.train_model \
    --processed_data_path "${DATA_PROCESSED_DIR}" \
    --model_output_path "${MODELS_DIR}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --run_name "${RUN_NAME}"

if [ $? -ne 0 ]; then
    echo "Error in Step 4: Train and Evaluate Model. Exiting."
    exit 1
fi
echo "Train and Evaluate Model - COMPLETED"
echo ""

# --- Pipeline Finished ---
echo "--------------------------------------"
echo "ML Pipeline Finished Successfully!"
echo "--------------------------------------"
echo "Output files (processed data, models) should be in the mounted volumes on your host."
echo "MLflow run data should be in the 'mlruns' directory on your host."
echo ""
echo "To view MLflow results:"
echo "1. Ensure this container (attrition_pipeline_executor) has finished."
echo "2. Run the MLflow UI service using: docker-compose up mlflow-ui"
echo "3. Open http://localhost:5001 (or your configured port) in your browser."
echo "--------------------------------------------------------------------"

exit 0