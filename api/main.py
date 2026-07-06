from fastapi import FastAPI
from pydantic import BaseModel, Field
from .routes import router

app = FastAPI(
    title="Health Insurance Prediction API",
    description="This API predicts whether a customer will buy health insurance or not based on their demographic and personal information.",
    version="1.0.0",
)

app.include_router(router)

"""
@app.post("/predict")
def predict(request: PredictionRequest):
   
    Accepts customer information for Health Insurance Cross Sell Prediction.
    
    from src.predict import Predictor
    predictor = Predictor()
    data = request.dict()
    prediction_result = predictor.predict_single(data)
    return {
        "id": data["id"],
        "prediction": prediction_result["prediction"],
        "probability": prediction_result["probability"],
        "label": prediction_result["label"]
    }
    return {
        "message": "Request received successfully",
        "data": request.model_dump()
    }"""