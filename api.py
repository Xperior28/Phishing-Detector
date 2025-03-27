from fastapi import FastAPI
import numpy as np
from tensorflow.keras.models import load_model
from pydantic import BaseModel
from feature_extractor import extract_url_features
from sklearn.preprocessing import StandardScaler
import joblib

# Initialize FastAPI app
app = FastAPI()

# Load the saved model
test_model = load_model("phishing_detection_model.h5")

# Load the saved scaler
scaler = joblib.load("scaler.pkl")

# Define input data structure
class URLRequest(BaseModel):
    url: str  # Expecting a single URL as input

@app.post("/predict")
async def predict_phishing(data: URLRequest):
    """
    API endpoint to predict phishing attacks based on input features.
    """
    # Extract features
    print(data.url)
    features = extract_url_features(data.url)
    features = np.array(features)
    features = features.reshape(1, -1)
    features = scaler.transform(features)

    # Make prediction
    print(features)
    prediction = test_model.predict(features)
    print(prediction)
    predicted_class = (prediction[:, 1] >= 0.5).astype(int)
    prediction = prediction.tolist()
    predicted_class = predicted_class.tolist()

    # Return the result
    return {
        "prediction":  predicted_class[0], 
        "score": prediction[0][1] * 100,
    }

