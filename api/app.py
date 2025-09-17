from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("model.pkl")

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict_tags(input: InputText):
    text = input.text
    y_pred = model.predict([text])
    predicted_tags = y_pred[0].tolist()
    return {"predicted_tags": predicted_tags}