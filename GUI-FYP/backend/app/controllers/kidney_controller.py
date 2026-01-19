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
@jwt_required()
@role_required(UserRole.DOCTOR)
def predict_kidney():
    

    input_data = request.get_json()
    prediction_result = predict_kidney_disease(input_data)

    return jsonify(prediction_result), 200
