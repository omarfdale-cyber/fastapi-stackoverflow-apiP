import joblib
import requests
import os
from fastapi import FastAPI
from pydantic import BaseModel

# URL correcte du modèle sur Hugging Face
MODEL_URL = "https://huggingface.co/OmarFD/stackoverflow-tagger-model/resolve/main/model.pkl"
MODEL_PATH = "model.pkl"

# Télécharger le modèle si nécessaire
if not os.path.exists(MODEL_PATH):
    print(f"Téléchargement du modèle depuis {MODEL_URL}...")
    response = requests.get(MODEL_URL)
    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
    else:
        raise RuntimeError(f"Impossible de télécharger le modèle depuis {MODEL_URL}")

# Charger le modèle
model = joblib.load(MODEL_PATH)

# Créer l'API
app = FastAPI(title="StackOverflow Tags Prediction API")

class TextInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l’API de prédiction de tags StackOverflow 🚀"}

@app.post("/predict")
def predict(input_data: TextInput):
    prediction = model.predict([input_data.text])
    return {"prediction": prediction.tolist()}
