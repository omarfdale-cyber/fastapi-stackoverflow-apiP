from fastapi import FastAPI, Request
from pydantic import BaseModel
from api.interface import load_model

app = FastAPI()
model = load_model()

class InputText(BaseModel):
    text: str

@app.post("/predict")
async def predict(input: InputText):
    prediction = model.predict([input.text])
    return prediction.tolist()