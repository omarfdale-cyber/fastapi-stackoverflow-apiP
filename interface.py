import streamlit as st
import joblib

# Chargement des objets nécessaires
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")
mlb = joblib.load("model/mlb.pkl")

st.title("Prédiction de tags StackOverflow")

question = st.text_area("Pose ta question ici")

if st.button("Prédire les tags"):
    vect = vectorizer.transform([question])
    pred = model.predict(vect)
    tags = mlb.inverse_transform(pred)
    st.write("Tags prédits :", tags[0])
