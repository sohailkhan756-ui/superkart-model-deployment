from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

superkart_api = Flask(__name__)
superkart_api.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

model = joblib.load(
    Path(__file__).with_name("superkart_model.joblib")
)
features = list(model.feature_names_in_)

numeric_features = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Age_Years"
]

def validate(frame):
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError("Missing fields: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("Provide at least one record.")

    frame = frame[features].copy()

    if frame.isna().any().any():
        raise ValueError("Missing values are not allowed.")

    for column in numeric_features:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise ValueError(column + " must contain finite numbers.")
        if (frame[column] < 0).any():
            raise ValueError(column + " cannot be negative.")

    if (frame["Product_Allocated_Area"] > 1).any():
        raise ValueError("Product_Allocated_Area must be between 0 and 1.")

    frame["Product_Sugar_Content"] = (
        frame["Product_Sugar_Content"].replace({"reg": "Regular"})
    )
    return frame

@superkart_api.errorhandler(ValueError)
def invalid_input(error):
    return jsonify({"error": str(error)}), 400

@superkart_api.after_request
def add_warning(response):
    response.headers["X-Model-Warning"] = (
        "Unseen categories, including Supermarket Type3, "
        "are not validated by training results."
    )
    return response

@superkart_api.get("/")
def home():
    return jsonify({"status": "ready", "service": "SuperKart API"})

@superkart_api.post("/v1/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("Provide a JSON object.")

    frame = validate(pd.DataFrame([payload]))
    prediction = model.predict(frame)[0]
    return jsonify({"Sales": round(float(prediction), 2)})

@superkart_api.post("/v1/predictbatch")
def predict_batch():
    if "file" not in request.files:
        raise ValueError("Upload a CSV using the file field.")

    try:
        frame = pd.read_csv(request.files["file"])
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
        raise ValueError("Unable to read the CSV file.")

    frame = validate(frame)
    predictions = model.predict(frame)

    return jsonify({
        str(i): round(float(value), 2)
        for i, value in enumerate(predictions)
    })

if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860)
