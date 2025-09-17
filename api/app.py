
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import requests

MODEL_URL = "https://huggingface.co/OmarFD/stackoverflow-tagger-model/resolve/main/model.pkl"
MODEL_PATH = "model.pkl"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
        else:
            raise Exception("Failed to download model")

download_model()
model = joblib.load(MODEL_PATH)

app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict_tags(input_text: InputText):
    try:
        tags = model.predict([input_text.text])
        return {"predicted_tags": tags[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
