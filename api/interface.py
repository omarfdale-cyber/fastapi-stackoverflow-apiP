import joblib
import os
import requests

def download_model():
    url = "https://huggingface.co/OmarFD/stackoverflow-tagger-model/resolve/main/model.pkl"
    response = requests.get(url)
    with open("model.pkl", "wb") as f:
        f.write(response.content)

def load_model():
    if not os.path.exists("model.pkl"):
        download_model()
    return joblib.load("model.pkl")