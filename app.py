from fastapi import FastAPI
from pydantic import BaseModel
from api.interface import load_model

app = FastAPI()

model = load_model()

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict_tags(data: InputText):
    prediction = model.predict([data.text])
    return prediction.tolist()
