from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)


model = joblib.load("models/crop_model.pkl")
le = joblib.load("models/label_encoder.pkl")
@app.route('/', methods=['GET', 'POST'])
def predict_crop():
    prediction = None
    if request.method == 'POST':
        try:
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])


            feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            input_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=feature_names)
            pred = model.predict(input_df)
            predicted_crop = le.inverse_transform(pred)[0]
            prediction = f"🌱 Recommended Crop: {predicted_crop}"
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)