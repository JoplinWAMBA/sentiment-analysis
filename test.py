import time
import pytest
from fastapi.testclient import TestClient
from main import app  

client = TestClient(app)


def test_predict_endpoint_valid():
    """Test l'endpoint /predict avec un tweet valide"""
    
    # Données de test
    test_data = {
        "text": "J'adore ce produit, il est fantastique !"
    }
    
    # Appel API
    response = client.post("/predict", json=test_data)
    
    # Validations critiques
    assert response.status_code == 200
    
    data = response.json()
    
    # Structure réponse
    required_fields = [
        "sentiment", "confidence", 
        "probability_positive", "probability_negative"
    ]
    for field in required_fields:
        assert field in data
    
    # Cohérence données
    assert data["sentiment"] in ["Positif", "Négatif"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["probability_positive"] <= 1.0
    assert 0.0 <= data["probability_negative"] <= 1.0
    
    # Somme probabilités = 1
    total_prob = data["probability_positive"] + data["probability_negative"]
    assert abs(total_prob - 1.0) < 0.01
    
    print(f"✅ Prediction OK: {data['sentiment']} ({data['confidence']:.2f})")



def test_predict_endpoint_invalid():
    """Test l'endpoint /predict avec des données invalides"""
    
    # Test 1 : Texte vide
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422
    
    # Test 2 : Texte trop long (> 280 caractères)
    long_text = "a" * 300
    response = client.post("/predict", json={"text": long_text})
    assert response.status_code == 422
    
    # Test 3 : Format JSON invalide
    response = client.post("/predict", json={})
    assert response.status_code == 422
    
    print("✅ Validation des erreurs OK")



def test_explain_endpoint():
    """Test l'endpoint /explain avec LIME"""
    
    # Données de test
    test_data = {
        "text": "Ce film est absolument terrible, je le déteste !"
    }
    
    # Mesure du temps (LIME peut être lent)
    start_time = time.time()
    response = client.post("/explain", json=test_data)
    duration = time.time() - start_time
    
    # Validations critiques
    assert response.status_code == 200
    
    data = response.json()
    
    # Structure réponse
    required_fields = [
        "sentiment", "explanation", "html_explanation"
    ]
    for field in required_fields:
        assert field in data
    
    # Validation explications
    assert isinstance(data["explanation"], list)
    assert len(data["explanation"]) > 0
    
    # Validation HTML LIME
    html_content = data["html_explanation"]
    assert isinstance(html_content, str)
    assert len(html_content) > 100  # HTML substantiel
    assert "<div" in html_content   # Contient du HTML
    
    # Performance acceptable (< 120 secondes)
    assert duration < 120
    
    print(f"✅ LIME OK: {len(data['explanation'])} mots expliqués")
    print(f"⏱️ Temps: {duration:.1f}s")



@pytest.mark.timeout(90)  # Timeout protection
def test_explain_robustness():
    """Test la robustesse de LIME avec différents textes"""
    
    test_cases = [
        "Super !",  # Texte très court
        "😊" * 10,  # Emojis uniquement
        "http://example.com test"  # Avec URLs
    ]
    
    for text in test_cases:
        response = client.post("/explain", json={"text": text})
        
        # Doit gérer tous les cas
        assert response.status_code in [200, 422]
        
        if response.status_code == 200:
            data = response.json()
            assert "html_explanation" in data
    
    print("✅ Robustesse LIME OK")