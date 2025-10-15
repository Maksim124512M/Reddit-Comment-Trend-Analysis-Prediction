from fastapi.testclient import TestClient
from main import app


client = TestClient(app)

def test_read_root():
    response = client.get('/predict/2027/')

    assert response.status_code == 200
    assert response.json() == {'year': 2027, 'predicted_comments': 9903}
