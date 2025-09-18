import os
import joblib
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# URL HuggingFace
MODEL_URL = "https://huggingface.co/OmarFD/stackoverflow-tagger-model/resolve/main/model.pkl"
MODEL_PATH = "model.pkl"

# Télécharger le modèle si pas déjà présent
if not os.path.exists(MODEL_PATH):
    print(f"Téléchargement du modèle depuis {MODEL_URL}...")
    r = requests.get(MODEL_URL)
    if r.status_code != 200:
        raise RuntimeError(f"Impossible de télécharger le modèle depuis {MODEL_URL}")
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

# Charger le modèle
model = joblib.load(MODEL_PATH)

# API FastAPI
app = FastAPI(title="StackOverflow Tags Predictor")

class Query(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de prédiction des tags StackOverflow !"}

@app.post("/predict")
def predict(query: Query):
    try:
        prediction = model.predict([query.text])
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
