from fastapi import FastAPI
from src.routes import endpoints

app = FastAPI(title="Intelligent Financial Fraud Detection System")


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Service is running"}


endpoints.setup_routes(app)
