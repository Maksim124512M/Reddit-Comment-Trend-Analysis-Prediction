import pickle
import numpy as np

from fastapi import FastAPI

app = FastAPI()

# Load pre-trained model
with open('linear_model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.get('/predict/{year}')
def predict(year: int):
    X_input = np.array([[year]])

    pred = model.predict(X_input)
    pred = max(pred[0], 0)

    return {'year': year, 'predicted_comments': int(pred)}