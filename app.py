
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Initialisation de l'app
app = FastAPI()

# Chargement du modèle et des classes
model = joblib.load("model/model.pkl")
classes = joblib.load("model/classes.pkl")

# Schéma de la requête attendue
class TextRequest(BaseModel):
    text: str
    threshold: float = 0.2

@app.post("/predict")
def predict(request: TextRequest):
    X = [request.text]
    proba = model.predict_proba(X)[0]

    print("\n🔍 Texte reçu :", request.text)
    print(f"🔧 Seuil utilisé : {request.threshold}")
    print("📊 Probabilités détectées > 0.01 :")
    for tag, p in zip(classes, proba):
        if p > 0.01:
            print(f"  - {tag}: {p:.3f}")

    predicted_tags = [tag for tag, p in zip(classes, proba) if p >= request.threshold]
    print("✅ Tags retenus :", predicted_tags)

    return {"predicted_tags": predicted_tags}
