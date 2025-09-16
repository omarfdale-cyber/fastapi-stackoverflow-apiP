
import os
import requests
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Téléchargement automatique du modèle 
MODEL_PATH = "model/model.pkl"
MODEL_URL = "https://huggingface.co/OmarFD/stackoverflow-tagger-model/resolve/main/model.pkl"

os.makedirs("model", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("Téléchargement du modèle depuis Hugging Face...")
    response = requests.get(MODEL_URL)
    if response.status_code != 200:
        raise Exception("Erreur lors du téléchargement du modèle")
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("Modèle téléchargé avec succès.")

# Chargement du modèle
model = joblib.load(MODEL_PATH)
print("Modèle chargé avec succès.")

# API
app = FastAPI()

class InputText(BaseModel):
    text: str
    threshold: float = 0.3

@app.post("/predict")
def predict_tags(input_data: InputText):
    try:
        X = [input_data.text]
        probs = model.predict_proba(X)[0]
        tags = model.classes_
        selected = [tag for tag, prob in zip(tags, probs) if prob >= input_data.threshold]
        return {"predicted_tags": selected}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
