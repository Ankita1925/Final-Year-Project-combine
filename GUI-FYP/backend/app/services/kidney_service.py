####################################################################
#
# File Name :   kidney_service.py
# Description : Kidney disease prediction service using VotingClassifier
# Author      : Pradhumnya Changdev Kalsait
# Date        : 19/01/26
#
####################################################################

import os
import numpy as np
import pandas as pd
import joblib
from flask import Blueprint, request, jsonify

kidney_blueprint = Blueprint("kidney", __name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(
    BASE_DIR, "ml_models", "kidney_voting_model.pkl"
)

model = joblib.load(MODEL_PATH)

FEATURE_ORDER = model.feature_names_in_

def predict_kidney_disease(input_data: dict) -> dict:
    df = pd.DataFrame([input_data])

    # Enforce schema strictly
    df = df[FEATURE_ORDER]

    stage = int(model.predict(df)[0])
    confidence = float(np.max(model.predict_proba(df)))

    stage_map = {
        0: ("No Kidney Disease", "LOW", "NO TRANSPLANT REQUIRED"),
        1: ("CKD Stage 1–2", "MEDIUM", "MEDICATION & MONITORING"),
        2: ("CKD Stage 3", "HIGH", "STRICT MONITORING"),
        3: ("CKD Stage 4", "VERY HIGH", "DIALYSIS REQUIRED"),
        4: ("End Stage Renal Disease", "CRITICAL", "TRANSPLANT REQUIRED"),
    }

    disease, criticality, decision = stage_map[stage]

    return {
        "organ": "KIDNEY",
        "stage": stage,
        "disease": disease,
        "criticality": criticality,
        "decision": decision,
        "confidence": f"{confidence * 100:.2f}%"
    }

@kidney_blueprint.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        return jsonify(predict_kidney_disease(data)), 200
    except Exception as e:
        print("Kidney prediction error:", e)
        return jsonify({"error": "Prediction failed"}), 500
