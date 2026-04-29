from flask import Blueprint, request, jsonify
from services.stage2_service import predict_stage2

stage2_bp = Blueprint("stage2", __name__)

@stage2_bp.route("/predict-stage2", methods=["POST"])
def stage2_predict():

    data = request.json

    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    try:
        result = predict_stage2(data)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500