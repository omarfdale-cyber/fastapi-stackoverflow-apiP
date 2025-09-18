from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import requests
import os

# ============================
# Config
# ============================
MODEL_URL = "https://huggingface.co/OmarFD/stackoverflow-tagger-model/resolve/main/model.pkl"
MODEL_PATH = "model.pkl"

# Liste des tags utilisés à l'entraînement
TAGS = [
    "python", "pandas", "numpy", "scikit-learn", "fastapi",
    "sql", "django", "regex", "api", "nlp"
]

# ============================
# Téléchargement du modèle
# ============================
if not os.path.exists(MODEL_PATH):
    print(f"Téléchargement du modèle depuis {MODEL_URL}...")
    r = requests.get(MODEL_URL)
    if r.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print(" Modèle téléchargé et sauvegardé.")
    else:
        raise RuntimeError(f"Impossible de télécharger le modèle depuis {MODEL_URL}")

# Charger le modèle
model = joblib.load(MODEL_PATH)

# ============================
# FastAPI
# ============================
app = FastAPI(title="StackOverflow Tags API")

class Item(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Bienvenue dans l'API de suggestion de tags StackOverflow 🚀"}

@app.post("/predict")
def predict(item: Item):
    # Prédiction brute (vecteur de 0/1)
    prediction = model.predict([item.text])[0]

    # Conversion en noms de tags
    tags_pred = [TAGS[i] for i, val in enumerate(prediction) if val == 1]

    return {"prediction": tags_pred}
