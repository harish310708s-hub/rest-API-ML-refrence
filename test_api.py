import pytest
from fastapi.testclient import TestClient

from app.main import app

VALID = [
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189,
]  # known malignant sample
BENIGN = [
    13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766,
    0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023,
    15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259,
]  # known benign sample


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # context manager runs the lifespan (loads the model)
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["n_features"] == 30 == len(body["feature_names"])
    assert body["classes"] == {"0": "malignant", "1": "benign"}


def test_predict_schema_and_status(client):
    r = client.post("/predict", json={"features": VALID})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"prediction", "label", "confidence", "probabilities"}
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("malignant", "benign")
    assert 0.0 <= body["confidence"] <= 1.0


def test_probabilities_sum_to_one(client):
    probs = client.post("/predict", json={"features": VALID}).json()["probabilities"]
    assert set(probs) == {"malignant", "benign"}
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)


def test_known_samples(client):
    assert client.post("/predict", json={"features": VALID}).json()["label"] == "malignant"
    assert client.post("/predict", json={"features": BENIGN}).json()["label"] == "benign"


def test_confidence_matches_predicted_class(client):
    body = client.post("/predict", json={"features": BENIGN}).json()
    assert body["confidence"] == pytest.approx(body["probabilities"][body["label"]])


def test_deterministic(client):
    a = client.post("/predict", json={"features": VALID}).json()
    b = client.post("/predict", json={"features": VALID}).json()
    assert a == b


@pytest.mark.parametrize("payload", [
    {},                                   # missing field
    {"features": VALID[:29]},             # too few
    {"features": VALID + [1.0]},          # too many
    {"features": ["a"] * 30},             # wrong type
    {"features": None},                   # null
    {"wrong_key": VALID},                 # wrong key
])
def test_invalid_payload_returns_422(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_non_finite_rejected(client):
    r = client.post("/predict", content='{"features": [NaN' + "," + ",".join(["1"] * 29) + "]}",
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_malformed_json_returns_422(client):
    r = client.post("/predict", content="not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_wrong_method_and_unknown_route(client):
    assert client.get("/predict").status_code == 405
    assert client.get("/nope").status_code == 404
