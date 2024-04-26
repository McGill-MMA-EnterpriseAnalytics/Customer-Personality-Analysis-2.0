from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import load_model, predict
import numpy as np

app = FastAPI(title="Prediction Model API")

# Load model at startup
model, scaler = load_model()

class PredictRequest(BaseModel):
    features: list

@app.post("/predict/")
async def get_prediction(request: PredictRequest):
    try:
        # Convert request to numpy array
        data = np.array([request.features])
        data_scaled = scaler.transform(data)  # Scale data
        prediction = predict(model, data_scaled)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
