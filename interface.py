import os
import joblib
import requests

def download_model_from_huggingface(url: str, local_path: str = "model/model.pkl"):
    os.makedirs("model", exist_ok=True)
    if not os.path.exists(local_path):
        print("🔄 Téléchargement du modèle depuis Hugging Face...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(response.content)
            print("✅ Modèle téléchargé avec succès.")
        else:
            raise Exception(f"❌ Erreur lors du téléchargement : {response.status_code}")

def load_model():
    hf_url = "https://huggingface.co/OMAR-FDALE/model-tag-prediction/resolve/main/model.pkl"
    local_path = "model/model.pkl"
    download_model_from_huggingface(hf_url, local_path)
    return joblib.load(local_path)
