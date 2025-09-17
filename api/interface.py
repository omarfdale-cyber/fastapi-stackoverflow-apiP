
import joblib

def load_model(path="model.pkl"):
    return joblib.load(path)

def predict_tags(model, text):
    return model.predict([text])[0]
