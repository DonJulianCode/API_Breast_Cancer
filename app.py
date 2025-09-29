from flask import Flask, request, jsonify
import joblib
import numpy as np

# Cargar modelo y scaler
model = joblib.load("models/breast_cancer_simple10_model.pkl")
scaler = joblib.load("models/simple10_scaler.pkl")
features = joblib.load("models/simple10_features.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "API de predicción de cáncer de mama (modelo simple 10 features)"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        input_values = [data[feat] for feat in features]
    except KeyError as e:
        return jsonify({"error": f"Falta la variable {str(e)}"}), 400

    X_new = np.array([input_values])
    X_scaled = scaler.transform(X_new)
    y_pred = model.predict(X_scaled)[0]
    y_proba = model.predict_proba(X_scaled)[0].max()

    result = {
        "prediction": "Maligno" if y_pred == 1 else "Benigno",
        "confidence": round(float(y_proba*100), 2)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
