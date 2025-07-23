import joblib
import numpy as np
import pandas as pd  # Add this line

# Load the trained model and label encoder
model = joblib.load("crop_model.pkl")
le = joblib.load("label_encoder.pkl")

N = float(input("N: "))
P = float(input("P: "))
K = float(input("K: "))
temperature = float(input("temperature: "))
humidity = float(input("humidity: "))
ph = float(input("ph: "))
rainfall = float(input("rainfall: "))

# Arrange input as model expects, using DataFrame with feature names
feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
input_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=feature_names)

# Predict
prediction = model.predict(input_df)

# Get crop name from label encoder
predicted_crop = le.inverse_transform(prediction)[0]

print(f"🌱 Recommended Crop: {predicted_crop}")