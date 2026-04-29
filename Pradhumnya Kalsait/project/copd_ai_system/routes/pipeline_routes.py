from flask import Blueprint, request, jsonify
from services.pipeline_service import full_copd_pipeline
import json

pipeline_bp = Blueprint("pipeline", __name__)

@pipeline_bp.route("/predict-full", methods=["POST"])
def predict_full():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    breath_file = request.files["file"]

    clinical_data = request.form.get("clinical_data")

    if clinical_data:
        clinical_data = json.loads(clinical_data)
    else:
        clinical_data = None

    try:
        result = full_copd_pipeline(breath_file, clinical_data)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500