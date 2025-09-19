import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# --- Config ---
MODEL_PATH = os.getenv("MODEL_PATH", "model/model.pkl")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))  # seuil pour activer une étiquette

app = FastAPI(title="StackOverflow Tags Predictor", version="0.1.0")

# --- Chargement modèle ---
bundle: Dict[str, Any] = {}
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)
    clf = bundle.get("model")
    mlb = bundle.get("mlb")
    if clf is None or mlb is None:
        raise ValueError("model.pkl must be a dict {'model': <estimator>, 'mlb': <MultiLabelBinarizer>}")
except Exception as e:
    # On garde l'API up, mais on signalera au /predict
    clf = None
    mlb = None
    load_error = str(e)
else:
    load_error = None

class PredictIn(BaseModel):
    text: str

@app.get("/")
def root():
    return {
        "status": "ok",
        "model_loaded": clf is not None,
        "classes_count": len(mlb.classes_) if mlb is not None else 0,
        "note": load_error if load_error else "ready",
    }

@app.get("/tags", response_model=List[str])
def tags():
    if mlb is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_error}")
    return [str(c) for c in mlb.classes_]

@app.post("/predict")
def predict(body: PredictIn):
    if clf is None or mlb is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_error}")

    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="Field 'text' must be a non-empty string")

    # Probabilités si dispo, sinon décision binaire
    y = None
    activated_indices: List[int] = []
    scores: List[float] = []

    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba([text])[0]  # shape (n_labels,)
        y = (probs >= THRESHOLD).astype(int).tolist()
        scores = [float(s) for s in probs]
        activated_indices = [i for i, p in enumerate(probs) if p >= THRESHOLD]
    else:
        preds = clf.predict([text])[0]  # binaire déjà
        y = [int(v) for v in preds]
        scores = y
        activated_indices = [i for i, v in enumerate(y) if v == 1]

    activated_tags = [str(mlb.classes_[i]) for i in activated_indices]

    # top 5 suggestions par score décroissant
    top5 = sorted(
        [{"label": str(mlb.classes_[i]), "score": float(scores[i]), "activated": (i in activated_indices), "index": i}
         for i in range(len(scores))],
        key=lambda x: x["score"],
        reverse=True
    )[:5]

    return {
        "activated_tags": activated_tags,
        "activated_indices": activated_indices,
        "top_suggestions": top5,
        "raw_prediction": y,
    }
