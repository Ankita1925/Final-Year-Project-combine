####################################################################
#
# File Name :   kidney_controller.py
# Description : kidney prediction API endpoints
# Author      : Pradhumnya Changdev Kalsait
# Date        : 17/01/26
#
####################################################################

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from app.services.kidney_service import predict_kidney_disease
from app.utils.jwt_utils import role_required
from app.utils.constants import UserRole

kidney_blueprint = Blueprint("kidney", __name__)

"""
################################################################
#
# Function Name : predict_kidney
# Description   : API endpoint for kidney disease prediction
# Author        : Pradhumnya Changdev Kalsait
# Date          : 17/01/26
# Prototype     : Response predict_kidney(void)
# Input Output  : (0 input, 1 output)
#
################################################################
"""
@kidney_blueprint.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("RECEIVED INPUT:", data)   # 👈 DEBUG

        if not data:
            return jsonify({"error": "Invalid JSON input"}), 400

        result = predict_kidney_disease(data)
        return jsonify(result), 200

    except Exception as e:
        print("❌ Kidney prediction error:", str(e))
        return jsonify({"error": str(e)}), 500
