
# StackOverflow Tag Suggestion API

Déployé via FastAPI sur Render. Télécharge automatiquement le modèle depuis Hugging Face.

## Endpoint

POST `/predict`

```json
{
  "text": "How to use a for loop in Python?"
}
```

Réponse attendue :
```json
{
  "predicted_tags": ["python", "for-loop"]
}
```

## Déploiement Render
- Python Web Service
- Port: 10000
- Start Command: `uvicorn app:app --host=0.0.0.0 --port=10000`
