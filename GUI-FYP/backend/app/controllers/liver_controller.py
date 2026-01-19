####################################################################
#
# File Name :   liver_controller.py
# Description : liver prediction API endpoints
# Author      : Pradhumnya Changdev Kalsait
# Date        : 17/01/26
#
####################################################################

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from app.services.liver_service import predict_liver_disease
from app.utils.jwt_utils import role_required
from app.utils.constants import UserRole

liver_blueprint = Blueprint("liver", __name__)

"""
################################################################
#
# Function Name : predict_liver
# Description   : API endpoint for liver disease prediction
# Author        : Pradhumnya Changdev Kalsait
# Date          : 17/01/26
# Prototype     : Response predict_liver(void)
# Input Output  : (0 input, 1 output)
#
################################################################
"""
@liver_blueprint.route("/predict", methods=["POST"])
@jwt_required()
@role_required(UserRole.DOCTOR)
def predict_liver():
    

    input_data = request.get_json()
    prediction_result = predict_liver_disease(input_data)

    return jsonify(prediction_result), 200
