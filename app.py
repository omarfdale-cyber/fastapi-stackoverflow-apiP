import os
import json
import joblib
import requests
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_URL = "https://huggingface.co/OmarFD/stackoverflow-tagger-model/resolve/main/model.pkl"
MODEL_PATH = "model.pkl"

# --------- Download model if missing ----------
if not os.path.exists(MODEL_PATH):
    print(f"[BOOT] Téléchargement du modèle depuis {MODEL_URL}...")
    r = requests.get(MODEL_URL)
    if r.status_code != 200:
        raise RuntimeError(f"Impossible de télécharger le modèle (HTTP {r.status_code})")
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("[BOOT] ✅ Modèle téléchargé.")

# --------- Load model ----------
print("[BOOT] Chargement du modèle...")
model = joblib.load(MODEL_PATH)
print("[BOOT] ✅ Modèle chargé.")

app = FastAPI(title="StackOverflow Tags Predictor")

class Query(BaseModel):
    text: str

# --------- Helpers ----------
def _safe_list(x):
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def guess_tag_names(m, n_classes):
    """
    Essaye de récupérer la liste des tags depuis le pipeline :
    - cherche un step avec .classes_ de longueur n_classes et de type string
    - sinon None (on utilisera "label_i")
    """
    # 1) Directement sur le modèle
    classes = getattr(m, "classes_", None)
    if classes is not None and len(classes) == n_classes:
        if len(classes) and isinstance(classes[0], (str, np.str_, bytes)):
            return [c.decode() if isinstance(c, bytes) else str(c) for c in classes]

    # 2) Sur un step du pipeline
    for attr in ("named_steps", "steps"):
        if hasattr(m, attr):
            steps = getattr(m, attr)
            # steps peut être dict (named_steps) ou liste (steps)
            iterable = steps.items() if isinstance(steps, dict) else steps
            for name, step in iterable:
                cls = getattr(step, "classes_", None)
                if cls is not None and len(cls) == n_classes:
                    if len(cls) and isinstance(cls[0], (str, np.str_, bytes)):
                        return [c.decode() if isinstance(c, bytes) else str(c) for c in cls]
                # Certains binarisers stockent les classes ailleurs
                for maybe in ("label_binarizer_", "binarizer_", "mlb_", "lb_"):
                    lb = getattr(step, maybe, None)
                    if lb is not None:
                        clb = getattr(lb, "classes_", None)
                        if clb is not None and len(clb) == n_classes:
                            if len(clb) and isinstance(clb[0], (str, np.str_, bytes)):
                                return [c.decode() if isinstance(c, bytes) else str(c) for c in clb]
    return None

def get_scores(m, X):
    """
    Retourne un vecteur de scores par classe (decision_function si possible,
    sinon predict_proba, sinon zeros).
    """
    if hasattr(m, "decision_function"):
        s = m.decision_function(X)
        s = s[0] if isinstance(s, (list, np.ndarray)) and np.ndim(s) > 1 else s
        return np.array(s, dtype=float)
    if hasattr(m, "predict_proba"):
        s = m.predict_proba(X)
        # en multilabel, predict_proba peut renvoyer une liste d'arrays → on uniformise
        if isinstance(s, list):
            # concat par colonne
            s = np.column_stack([np.array(si)[:, 1] if si.shape[1] == 2 else np.array(si).ravel() for si in s])
        s = s[0] if isinstance(s, (list, np.ndarray)) and np.ndim(s) > 1 else s
        return np.array(s, dtype=float)
    # fallback
    return None

@app.get("/")
def root():
    return {"message": "API de prédiction des tags StackOverflow prête 🚀"}

@app.post("/predict")
def predict(q: Query):
    try:
        X = [q.text]
        y = model.predict(X)
        # y → vecteur binaire multilabel (shape: (1, n_classes))
        y_vec = y[0] if isinstance(y, (list, np.ndarray)) and np.ndim(y) > 1 else y
        y_vec = np.array(y_vec).ravel()

        n_classes = y_vec.shape[0]
        tags = guess_tag_names(model, n_classes)

        # indices activés (==1)
        idx_ones = np.where(y_vec == 1)[0].tolist()
        activated = [ (tags[i] if tags else f"label_{i}") for i in idx_ones ]

        # récupérer des scores pour TOP-K suggestions
        scores = get_scores(model, X)
        suggestions = []
        if scores is not None and scores.shape[0] == n_classes:
            top_k = min(5, n_classes)
            top_idx = np.argsort(scores)[-top_k:][::-1]
            for i in top_idx:
                suggestions.append({
                    "label": (tags[i] if tags else f"label_{i}"),
                    "score": float(scores[i]),
                    "activated": bool(y_vec[i] == 1),
                    "index": int(i)
                })

        # Logs debug (visibles dans Render → Logs)
        print("[DEBUG] n_classes:", n_classes, flush=True)
        print("[DEBUG] activated_idx:", idx_ones, flush=True)
        print("[DEBUG] tags_found:", bool(tags), flush=True)
        if scores is not None:
            print("[DEBUG] sample_scores_top5:", json.dumps(suggestions, ensure_ascii=False), flush=True)

        return {
            "activated_tags": activated,
            "activated_indices": idx_ones,
            "top_suggestions": suggestions,
            "raw_prediction": y_vec.astype(int).tolist()  # pour diagnostiquer
        }

    except Exception as e:
        # remonter proprement l'erreur
        raise HTTPException(status_code=500, detail=str(e))
