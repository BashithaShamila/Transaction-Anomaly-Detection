# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies including the specific scikit-learn version
RUN pip install --no-cache-dir -r requirements.txt 

# Copy only the necessary directories and files
# This excludes .git folders, .gitignore, and markdown files
COPY src/ ./src/
COPY models/ ./models/
COPY tests/ ./tests/

# If you have any configuration files that are needed, copy them explicitly
# For example:
# COPY config.json .
# COPY .env.example .env

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]