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
@role_required(UserRole.DOCTOR)
def predict_lung():
    

    input_data = request.get_json()
    print("INPUT DATA:", input_data)

    prediction_result = predict_lung_disease(input_data)

    return jsonify(prediction_result), 200
