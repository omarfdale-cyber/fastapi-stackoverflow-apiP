# Fiche technique – Projet StackOverflow NLP

## Objectif
Prédire automatiquement les tags StackOverflow à partir du titre + corps d’une question, pour accélérer l’annotation et améliorer la recherche.

---

## Architecture du projet
- **Notebook 1 – Exploration & Pré-traitement** : nettoyage, analyse univariée/multivariée, création de features (`len_title`, `len_body`, etc.), export au format Parquet.
- **Notebook 2 – Requête API** : récupération des questions via Stack Exchange API / StackAPI (50 par requête), stockage dans un DataFrame.
- **Notebook 3 – Approche non supervisée (LDA)** : extraction de topics et visualisation en 2D.
- **Notebook 4 – Approche supervisée (MLflow)** : TF-IDF + OneVsRest(LogisticRegression), réduction aux 500 tags les plus fréquents, suivi des expérimentations avec MLflow.
- **API FastAPI** : endpoint `/predict` pour servir le modèle.
- **Déploiement** : Render (build automatique déclenché par push GitHub).

---

## Données
- **Source** : Stack Exchange API (questions StackOverflow).
- **Format** : fichiers CSV/Parquet.
- **Variables principales** : titre, corps, tags, score, nombre de vues, favoris, réponses, longueurs (`len_*`).
- **Filtrage** : conservation des 500 tags les plus fréquents.

---

## Modélisation
- **Vectorisation** : TF-IDF.
- **Modèle supervisé** : Logistic Regression (OneVsRestClassifier).
- **Binarisation des labels** : MultiLabelBinarizer.
- **Réduction** : top 500 tags pour limiter la mémoire.
- **Évaluation** : F1-score (micro).

---

## MLOps
- **Tracking** : MLflow (paramètres, métriques, modèles sauvegardés).
- **Versioning** : Git + GitHub (commits cohérents).
- **CI/CD** : déploiement continu via Render.
- **Stockage modèle** : Git LFS (`model/model.pkl`).

---

## Livrables attendus
- 4 notebooks (exploration, API, LDA, supervisé).
- Code API FastAPI (`/api`).
- Modèle entraîné (`model/model.pkl`).
- `README.md` explicatif.
- **Fiche technique (`fiche_technique_projet.md`)**.
- Support de présentation (15–25 slides).

