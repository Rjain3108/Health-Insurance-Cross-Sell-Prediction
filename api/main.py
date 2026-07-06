from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Health Insurance Prediction API",
    description="This API predicts whether a customer will buy health insurance or not based on their demographic and personal information.",
    version="1.0.0",
)

class PredictionRequest(BaseModel):
    id: int = Field(..., example=1)
    Gender: str = Field(..., example="Male")
    Age: int = Field(..., ge=18, le=100, example=35)
    Driving_License: int = Field(..., example=1)
    Region_Code: float = Field(..., example=28.0)
    Previously_Insured: int = Field(..., example=0)
    Vehicle_Age: str = Field(..., example="1-2 Year")
    Vehicle_Damage: str = Field(..., example="Yes")
    Annual_Premium: float = Field(..., example=40454.0)
    Policy_Sales_Channel: float = Field(..., example=26.0)
    Vintage: int = Field(..., example=217)

@app.get("/")
def home():
    return {"message": "Health Insurance Prediction API running. Use /docs for API documentation."}

@app.get("/ping")
def ping():
    return {"message": "Health Insurance Prediction API working", "status": "success", "code": 200, "version": "1.0.0"}

@app.post("/predict")
def predict(request: PredictionRequest):
    """
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
    }"""
    return {
        "message": "Request received successfully",
        "data": request.model_dump()
    }