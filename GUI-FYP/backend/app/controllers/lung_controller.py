####################################################################
#
# File Name :   lung_controller.py
# Description : Lung prediction API endpoints
# Author      : Pradhumnya Changdev Kalsait
# Date        : 17/01/26
#
####################################################################

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from app.services.lung_service import predict_lung_disease
from app.utils.jwt_utils import role_required
from app.utils.constants import UserRole
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
# -------------------------------------------------
# Blueprint
lung_blueprint = Blueprint("lung", __name__)

"""
################################################################
#
# Function Name : predict_lung
# Description   : API endpoint for lung disease prediction
# Author        : Pradhumnya Changdev Kalsait
# Date          : 17/01/26
# Prototype     : Response predict_lung(void)
# Input Output  : (0 input, 1 output)
#
################################################################
"""
@lung_blueprint.route("/predict", methods=["POST"])
@jwt_required()   
def predict():
    print("JWT IDENTITY:", get_jwt_identity())
    print("JWT CLAIMS:", get_jwt())

    try:
        if "image" not in request.files:
            return jsonify({"error": "Image file missing"}), 400

        image_file = request.files["image"]
        result = predict_lung_disease(image_file)
        return jsonify(result), 200

    except Exception as e:
        print("Lung prediction error:", e)
        return jsonify({"error": "Prediction failed"}), 500
