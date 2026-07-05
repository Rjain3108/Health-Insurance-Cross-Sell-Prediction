from fastapi import FastAPI

app = FastAPI(
    title="Health Insurance Prediction API",
    description="This API predicts whether a customer will buy health insurance or not based on their demographic and personal information.",
    version="1.0.0",
)

@app.get("/")
def home():
    return {"message": "Health Insurance Prediction API running. Use /docs for API documentation."}

@app.get("/ping")
def ping():
    return {"message": "Health Insurance Prediction API working", "status": "success", "code": 200, "version": "1.0.0"}