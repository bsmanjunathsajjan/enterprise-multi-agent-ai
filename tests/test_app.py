from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_run_workflow():
    response = client.post("/run", json={"task": "Plan a student event"})
    assert response.status_code == 200
    assert len(response.json()["agents"]) == 4
