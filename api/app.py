from fastapi import FastAPI
from pydantic import BaseModel
import os
import joblib
import urllib.request

app = FastAPI()

# Si model.pkl n'existe pas, le télécharger depuis Hugging Face
MODEL_URL = "https://huggingface.co/OmarFD/stackoverflow-tagger-model/resolve/main/model.pkl"
MODEL_LOCAL_PATH = "model.pkl"

if not os.path.exists(MODEL_LOCAL_PATH):
    print("Téléchargement du modèle depuis Hugging Face…")
    urllib.request.urlretrieve(MODEL_URL, MODEL_LOCAL_PATH)
    print("Modèle téléchargé.")

# Chargement du modèle
model = joblib.load(MODEL_LOCAL_PATH)

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict_tags(input: InputText):
    text = input.text
    y_pred = model.predict([text])
    predicted_tags = y_pred[0].tolist() if hasattr(y_pred[0], "tolist") else list(y_pred[0])
    return {"predicted_tags": predicted_tags}
