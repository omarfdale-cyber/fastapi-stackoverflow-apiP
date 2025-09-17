from transformers import pipeline
from huggingface_hub import hf_hub_download
import joblib

def multilabel_pipeline():
    model_path = hf_hub_download(repo_id="omarfdale/stack_tags_model", filename="model.pkl")
    model = joblib.load(model_path)
    return model