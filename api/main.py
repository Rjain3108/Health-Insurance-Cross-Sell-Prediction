from fastapi import FastAPI
from .routes import router

app = FastAPI(
    title="Health Insurance Prediction API",
    description="This API predicts whether a customer will buy health insurance or not based on their demographic and personal information.",
    version="1.0.0",
)

app.include_router(router)