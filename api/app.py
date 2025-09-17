
from fastapi import FastAPI, Request
from pydantic import BaseModel
from interface import multilabel_pipeline
from huggingface_hub import hf_hub_download

import os
import joblib

# Classe de requête
class Question(BaseModel):
    text: str

# Création de l'application FastAPI
app = FastAPI()

# Téléchargement automatique du modèle
if not os.path.exists("model"):
    os.makedirs("model")

MODEL_PATH = hf_hub_download(repo_id="omarfdale/stackoverflow-model",
                             filename="model.pkl",
                             cache_dir="model",
                             local_dir="model",
                             local_dir_use_symlinks=False)

# Chargement du modèle et des classes
model, mlb, vectorizer = joblib.load(MODEL_PATH)

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de suggestion de tags StackOverflow."}

@app.post("/predict")
def predict_tags(question: Question):
    try:
        text = question.text
        prediction = multilabel_pipeline(text, model, mlb, vectorizer)
        return {"tags": prediction}
    except Exception as e:
        return {"error": str(e)}
