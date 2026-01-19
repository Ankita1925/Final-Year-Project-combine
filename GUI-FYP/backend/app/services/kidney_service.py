####################################################################
#
# File Name :   kidney_service.py
# Description : Dummy ML prediction service for kidney disease (Flask)
# Author      : Pradhumnya Changdev Kalsait
# Date        : 17/01/26
#
####################################################################

from flask import Blueprint, request, jsonify
import pandas as pd
import joblib
import numpy as np

# -------------------------------------------------
# Blueprint
# -------------------------------------------------
kidney_blueprint = Blueprint("kidney", __name__)

# -------------------------------------------------
# Load trained model
# -------------------------------------------------
model = joblib.load("kidney_voting_model.pkl")

# -------------------------------------------------
# Prediction Logic
# -------------------------------------------------
def predict_kidney_disease(input_data: dict):
    df = pd.DataFrame([input_data])

    stage = model.predict(df)[0]
    confidence = np.max(model.predict_proba(df))

    stage_map = {
        0: ("No Kidney Disease", "LOW", "NO TRANSPLANT REQUIRED"),
        1: ("Chronic Kidney Disease (Stage 1–2)", "MEDIUM", "MEDICATION & MONITORING"),
        2: ("Chronic Kidney Disease (Stage 3)", "HIGH", "STRICT MONITORING"),
        3: ("Chronic Kidney Disease (Stage 4)", "VERY HIGH", "DIALYSIS REQUIRED"),
        4: ("End Stage Renal Disease", "CRITICAL", "TRANSPLANT REQUIRED")
    }

    disease, criticality, decision = stage_map[int(stage)]

    return {
        "organ": "KIDNEY",
        "disease": disease,
        "criticality": criticality,
        "decision": decision,
        "confidence": f"{confidence * 100:.2f}%"
    }

# -------------------------------------------------
# API Endpoint
# -------------------------------------------------
@kidney_blueprint.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Invalid JSON input"}), 400

        result = predict_kidney_disease(data)
        return jsonify(result), 200

    except Exception as e:
        print("Kidney prediction error:", e)
        return jsonify({"error": "Prediction failed"}), 500
