from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to FastAPI!"}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"


def test_read_items():
    response = client.get("/api/v1/items/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_create_item():
    test_item = {"name": "Test Item", "description": "Test Description"}
    response = client.post("/api/v1/items/", json=test_item)
    assert response.status_code == 200
    assert response.json()["name"] == test_item["name"]
    assert "id" in response.json()
