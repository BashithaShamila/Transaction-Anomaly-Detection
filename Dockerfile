# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies including the specific scikit-learn version
RUN pip install --no-cache-dir -r requirements.txt 

# Copy all application files
COPY src/ ./src/
COPY models/ ./models/
COPY tests/ ./tests/

# Just run the prediction script
CMD ["python", "-m", "src.predict_anomaly"]