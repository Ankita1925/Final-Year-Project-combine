####################################################################
#
# File Name :   heart_controller.py
# Description : heart prediction API endpoints
# Author      : Pradhumnya Changdev Kalsait
# Date        : 17/01/26
#
####################################################################

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from app.services.heart_service import predict_heart_disease
from app.utils.jwt_utils import role_required
from app.utils.constants import UserRole

heart_blueprint = Blueprint("heart", __name__)

"""
################################################################
#
# Function Name : predict_heart
# Description   : API endpoint for heart disease prediction
# Author        : Pradhumnya Changdev Kalsait
# Date          : 17/01/26
# Prototype     : Response predict_heart(void)
# Input Output  : (0 input, 1 output)
#
################################################################
"""
@heart_blueprint.route("/predict", methods=["POST"])
@jwt_required()
@role_required(UserRole.DOCTOR)
def predict_heart():
    

    input_data = request.get_json()
    prediction_result = predict_heart_disease(input_data)

    return jsonify(prediction_result), 200
