from flask import Blueprint, request, jsonify
from services.stage1_service import predict_stage1

stage1_bp = Blueprint("stage1", __name__)

@stage1_bp.route("/predict-stage1", methods=["POST"])
def stage1_predict():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        result = predict_stage1(file)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500