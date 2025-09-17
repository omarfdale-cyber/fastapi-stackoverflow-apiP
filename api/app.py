
from fastapi import FastAPI
from pydantic import BaseModel
from interface import load_model, predict_tags

class InputText(BaseModel):
    text: str

app = FastAPI()
model = load_model()

@app.post("/predict")
def predict(input: InputText):
    tags = predict_tags(model, input.text)
    return tags
