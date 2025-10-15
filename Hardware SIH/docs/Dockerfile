# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for the enhanced system
RUN pip install --no-cache-dir xgboost lightgbm flask fastapi uvicorn

# Copy all files
COPY . .

# Expose ports
EXPOSE 8501 5000

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]