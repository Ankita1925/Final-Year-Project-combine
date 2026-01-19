####################################################################
#
# File Name :   lung_service.py
# Description : Lung disease prediction service using CNN model
# Author      : Pradhumnya Changdev Kalsait
# Date        : 18/01/26
#
####################################################################

import os
from io import BytesIO
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -------------------------------------------------
# Blueprint
# -------------------------------------------------
lung_blueprint = Blueprint("lung", __name__)

# -------------------------------------------------
# Model configuration
# -------------------------------------------------
IMAGE_SIZE = 150
LABELS = [
    "Bacterial Pneumonia",
    "Corona Virus Disease",
    "Normal",
    "Tuberculosis",
    "Viral Pneumonia",
]

# -------------------------------------------------
# Load trained model (ABSOLUTE PATH)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "ml_models",
    "lung_disease_detection_model.h5"
)

lung_model = load_model(MODEL_PATH)

# -------------------------------------------------
# Prediction Logic
# -------------------------------------------------
################################################################
#
# Function Name : predict_lung_disease
# Description   : Predicts lung disease from chest X-ray image
# Author        : Pradhumnya Changdev Kalsait
# Date          : 18/01/26
# Prototype     : dict predict_lung_disease(File)
# Input Output  : (1 input, 1 output)
#
################################################################
def predict_lung_disease(image_file):

    # Read image bytes safely
    image_bytes = BytesIO(image_file.read())

    # Load and resize image
    img = load_img(image_bytes, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img)

    # IMPORTANT: DO NOT NORMALIZE (match training pipeline)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = lung_model.predict(img_array, verbose=0)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index])

    disease = LABELS[predicted_index]

    # Rule-based criticality mapping (NON-AI)
    criticality_map = {
        "Normal": ("LOW", "NO TREATMENT REQUIRED"),
        "Bacterial Pneumonia": ("HIGH", "HOSPITALIZATION REQUIRED"),
        "Viral Pneumonia": ("MEDIUM", "MEDICATION & MONITORING"),
        "Corona Virus Disease": ("HIGH", "ISOLATION & OXYGEN SUPPORT"),
        "Tuberculosis": ("CRITICAL", "LONG-TERM TREATMENT"),
    }

    criticality, decision = criticality_map.get(
        disease, ("UNKNOWN", "CONSULT SPECIALIST")
    )

    return {
        "organ": "LUNG",
        "disease": disease,
        "criticality": criticality,
        "decision": decision,
        "confidence": f"{confidence * 100:.2f}%",
    }

# -------------------------------------------------
# API Endpoint
# -------------------------------------------------
@lung_blueprint.route("/predict", methods=["POST"])
@jwt_required()
def predict():

    claims = get_jwt()

    if claims.get("role") != "DOCTOR":
        return jsonify({"error": "Doctor access only"}), 403

    if "image" not in request.files:
        return jsonify({"error": "Image file missing"}), 400

    image_file = request.files["image"]

    try:
        result = predict_lung_disease(image_file)
        return jsonify(result), 200
    except Exception as e:
        print("Lung prediction error:", e)
        return jsonify({"error": "Prediction failed"}), 500
