# Contributing to TransactionAnomalyDetection

## Branching Strategy

-   Use `feature/<short-description>` for new features.
    -   Example: `feature/anomaly-detector`
-   Use `fix/<short-description>` for bug fixes.
    -   Example: `fix/model-load-error`
-   Always create a pull request into `main` or `dev` branch with a clear description.

## Project Structure

```
src/
  controller/
    __init__.py       # Controller package init
    handler.py        # Service logic
  model/
    __init__.py       # Model package init
    fraud_model.py    # ML model loading
  routes/
    __init__.py       # Routes package init
    endpoints.py      # API endpoints
  main.py             # App entrypoint

tests/
  test_ping.py        # FastAPI route test

docs/
  # API specs, design documents, etc.

config/
  # Kafka, DB, or .env files

.github/workflows/
  ci.yml              # GitHub Actions pipeline

Dockerfile
requirements.txt
README.md
CONTRIBUTING.md
```

## Getting Started

```bash
git clone https://github.com/Financial-Fraud-Detection-System/transaction-anomaly-detection.git
cd transaction-anomaly-detection
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the App

```bash
uvicorn src.main:app --reload
```

## Run Tests

```bash
pytest tests/
```

## Code Formatting

We use [black](https://black.readthedocs.io/) for Python code formatting.

To format code:

```bash
black .
```

## Run using Docker

### Build the Docker image

```bash
docker build -t transaction-anomaly-detection .
```

### Run the container

```bash
docker run -p 8000:8000 transaction-anomaly-detection
```

Happy coding! ✨
