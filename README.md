# 💬 Twitter Sentiment Analyzer — API FastAPI

## 🧠 Description

Cette API fournit un service d’analyse de sentiments à partir de tweets, construite dans le cadre du projet **MLOps de 5A ESIEA**.  
Elle expose un modèle de **Machine Learning (Logistic Regression)** entraîné sur le dataset **Sentiment140**, et intègre **LIME** pour l’explicabilité locale des prédictions.

L’API est consommée par une interface **Streamlit** interactive, permettant à l’utilisateur de tester des tweets en temps réel et de visualiser les explications du modèle.

---

## 🚀 Fonctionnalités principales

**🔮 Prédiction de sentiment (`/predict`)**  
- Retourne la polarité du tweet (**Positif/Négatif**) et les probabilités associées.

**🧩 Explicabilité LIME (`/explain`)**  
- Fournit une visualisation HTML et un tableau des mots influents.

**❤️ Surveillance (`/health`)**  
- Vérifie que le modèle et le serveur fonctionnent correctement.

**🏠 Endpoint racine (`/`)**  
- Donne des informations générales sur l’API et le modèle chargé.

---

## 🧰 Stack technique

| Composant | Usage |
|-----------|-------|
| FastAPI   | Serveur d’API REST |
| Joblib    | Chargement du modèle scikit-learn |
| LIME      | Génération d’explications locales |
| Streamlit | Interface utilisateur interactive |
| Plotly    | Visualisation des probabilités |
| pytest    | Tests unitaires de l’API |

---