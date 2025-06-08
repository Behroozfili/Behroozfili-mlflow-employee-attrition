# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir: Disables the cache to reduce image size
# --trusted-host pypi.python.org: Can be useful if there are network issues with PyPI SSL
RUN pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Make port 5000 available to the world outside this container (for MLflow UI)
EXPOSE 5000
# Make port 8080 available if you plan to serve a model using MLflow serve
EXPOSE 8080


# Define environment variable (optional, can be useful)
ENV APP_HOME /app
ENV PYTHONPATH "${APP_HOME}:${PYTHONPATH}"

# Command to run when the container launches.
# You can choose to run the shell script or the python script.
# Example 1: Run the training shell script
# CMD ["bash", "scripts/run_training.sh"]

# Example 2: Run the training python script
CMD ["python", "scripts/run_experiment.py"]

# Example 3: Or, just start the MLflow UI (if training is done outside or in a previous step)
# CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"]