fimport os
import joblib
import requests
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# ===============================
# Configuration
# ===============================
MODEL_URL = "https://huggingface.co/OmarFD/stackoverflow-tagger-model/resolve/main/model.pkl"
MODEL_PATH = "model.pkl"

# ===============================
# Téléchargement du modèle
# ===============================
if not os.path.exists(MODEL_PATH):
    print(f"Téléchargement du modèle depuis {MODEL_URL}...")
    response = requests.get(MODEL_URL)
    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("✅ Modèle téléchargé avec succès.")
    else:
        raise RuntimeError(f"❌ Impossible de télécharger le modèle depuis {MODEL_URL}")

# ===============================
# Chargement du modèle
# ===============================
print("Chargement du modèle...")
model = joblib.load(MODEL_PATH)
print("✅ Modèle chargé avec succès.")

# ===============================
# API FastAPI
# ===============================
app = FastAPI(title="StackOverflow Tags Predictor")

class Query(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de prédiction des tags StackOverflow 🚀"}

@app.post("/predict")
def predict(query: Query):
    text = query.text

    try:
        # Prédiction brute
        prediction = model.predict([text])[0]

        # Conversion en format JSON-compatible
        if isinstance(prediction, (np.ndarray, list, tuple)):
            prediction = [str(p) for p in prediction]
        else:
            prediction = str(prediction)

        return {"prediction": prediction}

    except Exception as e:
        return {"error": str(e)}
