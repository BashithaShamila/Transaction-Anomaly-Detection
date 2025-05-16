# Transaction Anomaly Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository is part of the Intelligent Financial Fraud Detection System project. It provides a machine learning-based solution for detecting anomalous financial transactions.

## 🚀 Features

- Isolation Forest algorithm for anomaly detection
- Deep learning autoencoder model as an additional detection method
- API endpoints for real-time transaction analysis
- Preprocessing pipeline for transaction data
- Containerized deployment with Docker

## 🛠️ Technologies Used

- Python 3.9
- TensorFlow/Keras
- scikit-learn
- FastAPI
- Docker

## 📋 Prerequisites

- Python 3.9 or higher
- Docker (for containerized deployment)
- Git

## 🔧 Installation and Setup

### Local Development

```bash
# Clone the repository
git clone https://github.com/Financial-Fraud-Detection-System/transaction-anomaly-detection.git

# Change to project directory
cd transaction-anomaly-detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Docker Deployment

```bash
# Build the Docker image
docker build -t transaction-anomaly-detection .

# Run the container
docker run -p 8000:8000 transaction-anomaly-detection
```

## 🚀 Usage

### Running the API Server

```bash
uvicorn src.main:app --reload
```

The API will be available at http://localhost:8000

### API Documentation

Once the server is running, you can access the interactive API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🧪 Testing

```bash
# Run all tests
pytest tests/
```

## 📁 Project Structure

```
src/
  controller/          # Service logic handlers
  model/               # ML model implementations
  routes/              # API endpoints
  main.py              # App entrypoint
  predict_anomaly.py   # Prediction service

models/                # Trained model files
  isolation_forest_model.pkl
  autoencoder_model_scaled.keras
  category_mappings.json
  scaler.pkl

tests/                 # Test files
```

## 🤝 Contributing

Contributions are welcome! Please check out our [contribution guidelines](CONTRIBUTING.md).

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

