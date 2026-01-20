####################################################################
#
# File Name :   heart_controller.py
# Description : Heart disease prediction API controller
# Author      : Pradhumnya Changdev Kalsait
# Date        : 20/01/26
#
####################################################################

from flask import Blueprint, request, jsonify
from app.services.heart_service import predict_heart_disease

heart_blueprint = Blueprint("heart_blueprint", __name__)

"""
################################################################
#
# Function Name : predict_heart
# Description   : Heart disease prediction REST API
# Author        : Pradhumnya Changdev Kalsait
# Date          : 20/01/26
# Prototype     : dict predict_heart(void)
# Input Output  : (1 input, 1 output)
#
################################################################
"""

@heart_blueprint.route("/predict", methods=["POST"])
def predict_heart():

    input_data = request.get_json(force=True)

    result = predict_heart_disease(input_data)

    return jsonify({
        "success": True,
        "result": {
            "organ": result["organ"],
            "disease_type": result["disease"],
            "risk_level": result["criticality"].title(),
            "medical_action": result["decision"],
            "presence": "Present" if "PRESENT" in result["criticality"] else "Absent"
        }
    }), 200
