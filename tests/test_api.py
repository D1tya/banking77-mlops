from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_root():

    response = client.get("/")

    print("ROOT RESPONSE:", response.status_code)
    print("ROOT BODY:", response.text)

    assert response.status_code == 200


def test_health():

    response = client.get("/health")

    print("HEALTH RESPONSE:", response.status_code)
    print("HEALTH BODY:", response.text)

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "LinearSVC"


def test_prediction():

    response = client.post(
        "/predict",
        json={
            "text": "My card hasn't arrived yet"
        },
    )

    print("PREDICTION STATUS:", response.status_code)
    print("PREDICTION BODY:", response.text)

    assert response.status_code == 200

    data = response.json()

    assert data["category"] == "card_arrival"
    assert "score" in data
    assert "top_predictions" in data
    assert len(data["top_predictions"]) == 3


def test_prediction_exchange():

    response = client.post(
        "/predict",
        json={
            "text": "I need to exchange currencies"
        },
    )

    print("EXCHANGE STATUS:", response.status_code)
    print("EXCHANGE BODY:", response.text)

    assert response.status_code == 200

    data = response.json()

    assert data["category"] == "exchange_via_app"


def test_empty_text():

    response = client.post(
        "/predict",
        json={
            "text": ""
        },
    )

    assert response.status_code == 422


def test_missing_text():

    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 422
