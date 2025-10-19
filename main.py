from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
##from app_fastapi import app
from pydantic import BaseModel, Field
from typing import List, Dict
import joblib
import re
import string
from lime.lime_text import LimeTextExplainer
import numpy as np
import time


# ---------- Request / Response models ----------
class TweetRequest(BaseModel):
    text: str = Field(..., max_length=280)

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    probability_positive: float
    probability_negative: float

class ExplanationResponse(BaseModel):
    sentiment: str
    explanation: List[Dict[str, float]]
    html_explanation: str


# ---------- App init ----------
app = FastAPI(
    title="Sentiment Analysis API",
    description="FastAPI wrapping an MLflow-logged sentiment model (Sentiment140). Endpoints: /predict, /explain, /health, /",
    version="0.1"
)


# Authorize local request
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Configuration of MLflow tracking URI ----------
#mlflow.set_tracking_uri("http://mlflow-server:5000") # à checker lors des tests

# Joblib strategy for optimal performance
try:
    model = joblib.load("./artifacts/sentiment_model.joblib")
    vectorizer = joblib.load("./artifacts/tfidf_vectorizer.joblib")
    lime_explainer = LimeTextExplainer(class_names=["negative", "positive"])
    print("✅ Modèles chargés avec succès")
except Exception as e:
    model, vectorizer, lime_explainer = None, None, None
    print(f"❌ Error occuring when loading model : {e}")


def preprocess_text(text: str) -> str:
    """
    Nettoie le texte pour le rendre compatible avec le modèle :
    - supprime URLs, mentions, hashtags
    - retire ponctuation et chiffres
    - met en minuscule
    """
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # URLs
    text = re.sub(r"@\w+|#", "", text)                   # mentions & hashtags
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))
    return text.strip()

# Endpoints

@app.get("/")
def root():
    """ Endpoint racine : retourne l'état de l'API """
    return {"status": "online", "model_loaded": model is not None}

@app.get("/health")
def health():
    """ Vérifie l'état du modèle """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")
    return {"status": "healthy", "model": "./artifacts/sentiment_model.joblib"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TweetRequest):
    """ Prédit le sentiment d’un tweet """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    text = preprocess_text(request.text)
    X = vectorizer.transform([text])
    proba = model.predict_proba(X)[0]
    sentiment = "Positif" if proba[1] > 0.5 else "Négatif"
    confidence = abs(proba[1] - 0.5) * 2  # niveau de confiance

    return PredictionResponse(
        sentiment=sentiment,
        confidence=float(confidence),
        probability_positive=float(proba[1]),
        probability_negative=float(proba[0])
    )


@app.post("/explain", response_model=ExplanationResponse)
def explain(request: TweetRequest):
    """ Génère une explication LIME pour un texte donné """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    text = preprocess_text(request.text)

    # Génération d'une explication LIME
    exp = lime_explainer.explain_instance(
        text_instance=text,
        classifier_fn=lambda x: model.predict_proba(vectorizer.transform(x)),
        num_features=10
    )

    explanation = [{"word": w, "weight": float(v)} for w, v in exp.as_list()]
    html_explanation = exp.as_html()

    # Prédiction pour récupérer le sentiment dominant
    X = vectorizer.transform([text])
    proba = model.predict_proba(X)[0]
    sentiment = "Positif" if proba[1] > 0.5 else "Négatif"

    return ExplanationResponse(
        sentiment=sentiment,
        explanation=explanation,
        html_explanation=html_explanation
    )
