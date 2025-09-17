from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from huggingface_hub import hf_hub_download
from interface import multilabel_pipeline

# Téléchargement du modèle depuis Hugging Face
model_path = hf_hub_download(repo_id="omarfdale/projetstackoverflow", filename="model.pkl")
model = joblib.load(model_path)

# Chargement de l'objet multi-label pipeline
mlb = model.named_steps['clf'].classes_

app = FastAPI()

class InputData(BaseModel):
    text: str
    threshold: float = 0.3

@app.get("/")
def home():
    return {"message": "API de suggestion de tags StackOverflow – Projet Omar"}

@app.post("/predict")
def predict_tags(data: InputData):
    try:
        text = data.text
        proba = model.predict_proba([text])[0]
        tags = (proba > data.threshold)
        return {"tags": list(mlb[tags])}
    except Exception as e:
        return {"error": str(e)}
