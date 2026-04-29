import joblib
import pandas as pd
import numpy as np
import json

MODEL_PATH = "models/ExtraTreesS2.pkl"
FEATURE_PATH = "models/stage2_features.json"

stage2_model = joblib.load(MODEL_PATH)

with open(FEATURE_PATH, "r") as f:
    required_features = json.load(f)

STAGE2_CONF_THRESHOLD = 0.50


def predict_stage2(input_data: dict):

    ordered_data = {f: input_data[f] for f in required_features}
    df = pd.DataFrame([ordered_data])

    prediction = stage2_model.predict(df)[0]
    probabilities = stage2_model.predict_proba(df)[0]

    confidence = float(np.max(probabilities))

    probability_breakdown = {
        f"GOLD_{i+1}": float(probabilities[i])
        for i in range(len(probabilities))
    }

    return {
        "gold_stage": int(prediction),
        "confidence": confidence,
        "probabilities": probability_breakdown,
        "meets_threshold": confidence >= STAGE2_CONF_THRESHOLD
    }