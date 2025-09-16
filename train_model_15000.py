
import pandas as pd
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# 1. Charger les données
df = pd.read_parquet("clean_questions.parquet")
df = df.sample(n=15000, random_state=42)  # plus de données

# 2. Séparer texte et tags
X = df["text"]
y = df["tag_list"]

# 3. Binarisation multilabel
mlb = MultiLabelBinarizer()
y_bin = mlb.fit_transform(y)

# 4. Pipeline avec TF-IDF amélioré et classifieur
pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9
    )),
    ("clf", OneVsRestClassifier(LogisticRegression(max_iter=1000), verbose=1))
])

# 5. Entraînement
pipeline.fit(X, y_bin)

# 6. Test rapide d'une phrase
test_text = ["How to fix a Python TypeError when using pandas groupby?"]
proba = pipeline.predict_proba(test_text)[0]
print("\n🔍 Tags détectés avec proba > 0.05 :")
for tag, p in zip(mlb.classes_, proba):
    if p > 0.05:
        print(f"{tag}: {p:.3f}")

# 7. Sauvegarde
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/model.pkl")
joblib.dump(mlb.classes_, "model/classes.pkl")
print("\n✅ Modèle entraîné et sauvegardé avec 15 000 lignes.")
