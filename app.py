import os
import joblib
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# URL du modèle hébergé sur Hugging Face (à adapter si besoin)
MODEL_URL = "https://huggingface.co/OMAR-FDALE/model-tag-prediction/resolve/main/model.pkl"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

app = FastAPI(title="StackOverflow Tags API", version="1.0.0")

# Téléchargement du modèle si non présent
os.makedirs(MODEL_DIR, exist_ok=True)
if not os.path.exists(MODEL_PATH):
    r = requests.get(MODEL_URL)
    if r.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
    else:
        raise RuntimeError(f"Impossible de télécharger le modèle depuis {MODEL_URL}")

# Chargement du modèle
model = joblib.load(MODEL_PATH)

class PredictIn(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "API StackOverflow Tags en ligne ✅"}

@app.post("/predict")
def predict(payload: PredictIn):
    try:
        prediction = model.predict([payload.text])
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
