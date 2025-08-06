from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load all models once
models = {
    'decision_tree': joblib.load('D:/Liver Disease Prediction Using Various Models/Models/Decision_tree.pkl'),
    'adaboost': joblib.load('D:/Liver Disease Prediction Using Various Models/Models/Adaboost.pkl'),
    'knn': joblib.load('D:/Liver Disease Prediction Using Various Models/Models/KNN.pkl'),
    'logistic_regression': joblib.load('D:/Liver Disease Prediction Using Various Models/Models/Logistic_regression.pkl'),
    'random_forest': joblib.load('D:/Liver Disease Prediction Using Various Models/Models/Random_forest.pkl'),
    'svc': joblib.load('D:/Liver Disease Prediction Using Various Models/Models/SVC.pkl'),
    'xgboost': joblib.load('D:/Liver Disease Prediction Using Various Models/Models/XGBClassifier.pkl')
}

@app.route('/predict', methods=['POST'])
def predict_selected():
    data = request.json

    # Extract selected models
    selected_models = data.get("selected_models", [])
    if not selected_models:
        return jsonify({"error": "No models selected."}), 400

    # Validate models
    for model_name in selected_models:
        if model_name not in models:
            return jsonify({"error": f"Invalid model: {model_name}"}), 400

    # Prepare input features
    input_data = {k: v for k, v in data.items() if k != "selected_models"}
    features = np.array([list(input_data.values())])

    # Make predictions with selected models
    results = {}
    for model_name in selected_models:
        model = models[model_name]
        prediction = model.predict(features)
        result = "Disease" if prediction[0] == 1 else "Non Disease"
        results[model_name] = result
        print(f"[INFO] {model_name}: {result}")

    return jsonify(results)

if __name__ == '__main__':
    print("[INFO] Running at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
