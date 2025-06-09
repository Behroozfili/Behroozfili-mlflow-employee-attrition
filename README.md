
# 🚀 Employee Attrition Prediction ML Pipeline

This project implements an **end-to-end MLOps pipeline** for predicting employee attrition. It includes steps for data loading, preprocessing, feature engineering, model training, evaluation, and experiment tracking using **MLflow**. The pipeline can be executed either **locally using Python** or inside **Docker containers using Docker Compose**.

---

## 📚 Table of Contents

- [✨ Features](#-features)
- [📁 Project Structure](#-project-structure)
- [🛠️ Prerequisites](#️-prerequisites)
- [⚙️ Setup](#️-setup)
  - [1. Clone Repository](#1-clone-repository)
  - [2. Prepare Data](#2-prepare-data)
  - [3. Python Environment (for Local Execution)](#3-python-environment-for-local-execution)
- [🚦 Running the Pipeline](#-running-the-pipeline)
  - [Option 1: Using Docker (Recommended)](#option-1-using-docker-recommended)
  - [Option 2: Running Locally with Python Orchestrator](#option-2-running-locally-with-python-orchestrator)
- [🔁 Pipeline Steps](#-pipeline-steps)
- [📦 Outputs](#-outputs)
- [🐞 Troubleshooting](#-troubleshooting)

---

## ✨ Features

- 🔄 **Data Loading & Preprocessing**  
- 🔀 **Train/Test Splitting**  
- 🧪 **Feature Engineering** (e.g., scaling, encoding)  
- 🤖 **Model Training & Evaluation** (Logistic Regression, Random Forest)  
- 📊 **MLflow Experiment Tracking**  
- 🐳 **Dockerized Execution**  
- 🐍 **Python-Based Orchestration**  

---

## 📁 Project Structure

.
├── data
│ ├── raw/ # Input raw data (e.g., employee_attrition.csv)
│ └── processed/ # Processed data (splits, transformed data)
├── models/ # Trained models and preprocessors
├── mlruns/ # MLflow tracking data
├── scripts/
│ └── run_training.sh # Bash script for Docker-based execution
├── src/
│ ├── data/
│ │ ├── load_data.py # Load and preprocess data
│ │ └── split.py # Data splitting
│ ├── features/
│ │ └── build_features.py # Feature engineering
│ ├── models/
│ │ └── train_model.py # Train and evaluate model
│ └── init.py
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── run_pipeline.py # Local orchestrator
└── README.md # This file

yaml
Copy
Edit

---

## 🛠️ Prerequisites

- **Git**
- **Docker & Docker Compose**
- **Python 3.8+**
- **pip**
- **Raw Dataset:** Place `employee_attrition.csv` into `data/raw/`.

---

## ⚙️ Setup

### 1. Clone Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
2. Prepare Data
bash
Copy
Edit
mkdir -p data/raw
Place your employee_attrition.csv file into data/raw/.

3. Python Environment (for Local Execution)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Create and populate requirements.txt:

txt
Copy
Edit
# requirements.txt
pandas
scikit-learn
joblib
mlflow
numpy
matplotlib
seaborn
Then install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
🚦 Running the Pipeline
✅ Option 1: Using Docker (Recommended)
🔧 Build and Run Pipeline
bash
Copy
Edit
docker-compose build run-pipeline
docker-compose up run-pipeline
Or build all services:

bash
Copy
Edit
docker-compose build
docker-compose up run-pipeline
🌐 Access MLflow UI (Docker)
bash
Copy
Edit
docker-compose up mlflow-ui
Navigate to http://localhost:5001 to view experiment logs.

🐍 Option 2: Running Locally with Python Orchestrator
▶️ Run Pipeline Script
bash
Copy
Edit
source venv/bin/activate
python run_pipeline.py
🌐 Access MLflow UI (Local)
bash
Copy
Edit
mlflow ui
Navigate to http://localhost:5000 to explore runs and metrics.

🔁 Pipeline Steps
Load and Preprocess Data

Input: data/raw/employee_attrition.csv

Output: processed_employee_attrition.csv

Split Data

Output: train.csv, test.csv

Build Features

Output: X_train_processed.csv, X_test_processed.csv, y_train.csv, y_test.csv

Artifacts: preprocessor.joblib

Train and Evaluate Model

Output: model.joblib

Logs: Parameters, metrics, plots (e.g., ROC, confusion matrix)

📦 Outputs
data/processed/

processed_employee_attrition.csv

train.csv, test.csv

X_train_processed.csv, X_test_processed.csv

y_train.csv, y_test.csv

models/

preprocessor.joblib

model.joblib

mlruns/

MLflow experiments, parameters, metrics, and artifacts

🐞 Troubleshooting
FileNotFoundError for employee_attrition.csv:
Ensure it's located in data/raw/.

Docker Volume Permission Issues (Linux):

bash
Copy
Edit
sudo chown -R $(id -u):$(id -g) data models mlruns
Port Conflicts (5000 or 5001):
Update ports in docker-compose.yml or use:

bash
Copy
Edit
mlflow ui --port 5050
Module Not Found (Local):

Activate your environment.

Run from the project root.

Ensure correct PYTHONPATH.

Module Not Found (Docker):
Dockerfile sets PYTHONPATH=/app. Ensure you’re using python -m from project root.

📌 Note: Replace placeholders like <your-repository-url> and <your-repository-name> before committing.

✅ To-Do Before Commit
✅ Add a complete requirements.txt

✅ Place employee_attrition.csv in data/raw/

✅ Add .gitignore to exclude:

venv/, __pycache__/, *.pyc

data/processed/, models/, mlruns/ (unless intentional)
Behrooz Filzadeh

Happy ML Ops-ing! 🚀









