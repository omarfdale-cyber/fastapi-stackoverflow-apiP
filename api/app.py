from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
import joblib
import os

# Téléchargement du modèle depuis Hugging Face
MODEL_PATH = hf_hub_download(
    repo_id="OmarFD/stackoverflow-tagger-model",
    filename="model.pkl",
    cache_dir="model",
    local_dir="model"
)

# Chargement du modèle
model = joblib.load(MODEL_PATH)

# API FastAPI
app = FastAPI()

class InputData(BaseModel):
    text: str

@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([data.text])
    return {"tags": prediction[0]}