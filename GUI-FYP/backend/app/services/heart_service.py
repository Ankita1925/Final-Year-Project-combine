####################################################################
#
# File Name :   heart_service.py
# Description : ML prediction service for heart disease
# Author      : Pradhumnya Changdev Kalsait
# Date        : 17/01/26
#
####################################################################

import os
import joblib
import pandas as pd
import numpy as np

# Global predictor instance to avoid reloading models on every call
_predictor = None

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_DIR = os.path.join(
    BASE_DIR,
    "ml_models",
    "models_heartcsv"
)
class HeartDiseasePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.imputer = None
        self.load_models()

    def load_models(self):
        """Load trained models and preprocessing objects"""
        try:
            # Load Scikit-learn models
            # Check if models directory exists
            if not os.path.exists(MODEL_DIR):
                # Try absolute path fallback
                abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models_heartcsv'))
                if os.path.exists(abs_path):
                    self.model_dir = abs_path
                else:
                    self.model_dir = MODEL_DIR # Keep original to show error
            else:
                self.model_dir = MODEL_DIR

            self.model = joblib.load(os.path.join(self.model_dir, 'best_model.pkl'))
            self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
            self.imputer = joblib.load(os.path.join(self.model_dir, 'imputer.pkl'))
        except Exception as e:
            print(f"Error loading models: {e}")
            self.model = None

    def preprocess_input(self, data):
        """Preprocess input data for prediction"""
        try:
            # Create DataFrame from input dictionary
            # Ensure columns match the training data exactly
            columns = [
                'Age', 'Gender', 'Height(cm)', 'Weight(kg)', 
                'Systolic_Blood_Pressure', 'Diastolic_Blood_Pressure', 
                'Cholesterol', 'Glucose', 'Smoking', 'Alcohol', 'Physical_Activity'
            ]
            
            # Create DataFrame with a single row
            df = pd.DataFrame([data], columns=columns)
            
            # Impute missing values (if any)
            df_imputed = pd.DataFrame(
                self.imputer.transform(df),
                columns=columns
            )
            
            # Scale features
            df_scaled = self.scaler.transform(df_imputed)
            
            return df_scaled, None
        except Exception as e:
            return None, str(e)

    def predict(self, data):
        """Make prediction using the loaded model"""
        if not self.model:
            return None, "Model not loaded"

        try:
            processed_data, error = self.preprocess_input(data)
            if error:
                return None, error

            # Get prediction probabilities
            if hasattr(self.model, 'predict_proba'):
                prediction_proba = self.model.predict_proba(processed_data)[0]
                prediction = np.argmax(prediction_proba)
                
                # Probability of disease (class 1)
                prob_disease = prediction_proba[1] * 100
            else:
                # For models without predict_proba
                prediction = self.model.predict(processed_data)[0]
                prob_disease = 100.0 if prediction == 1 else 0.0

            # Map prediction to disease type and risk info
            # 0: Absence, 1: Presence
            
            if prediction == 0:
                result_info = {
                    "type": "No Cardiovascular Disease",
                    "risk": "Low Risk",
                    "presence": "Absent",
                    "recommendation": "Maintain a healthy lifestyle. Regular annual checkups recommended.",
                    "action": "Preventive Care"
                }
            else:
                # Determine severity based on probability if available
                if prob_disease < 60:
                    result_info = {
                        "type": "Cardiovascular Disease (Mild Risk)",
                        "risk": "Moderate Risk",
                        "presence": "Present",
                        "recommendation": "Lifestyle modifications (diet, exercise). Consult cardiologist.",
                        "action": "Medical Management"
                    }
                elif prob_disease < 80:
                    result_info = {
                        "type": "Cardiovascular Disease (Moderate Risk)",
                        "risk": "High Risk",
                        "presence": "Present",
                        "recommendation": "Strict medical therapy required. Further diagnostic tests recommended.",
                        "action": "Intervention Consideration"
                    }
                else:
                    result_info = {
                        "type": "Cardiovascular Disease (Severe Risk)",
                        "risk": "Critical Risk",
                        "presence": "Present",
                        "recommendation": "Immediate medical attention required. Evaluation for advanced therapies.",
                        "action": "Urgent Care"
                    }

            return {
                'disease_type': result_info['type'],
                'risk_level': result_info['risk'],
                'medical_action': result_info['action'],
                'recommendation': result_info['recommendation']
            }, None

        except Exception as e:
            return None, str(e)

def _get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = HeartDiseasePredictor()
    return _predictor

"""
################################################################
#
# Function Name : predict_heart_disease
# Description   : Simulates heart disease and criticality prediction
# Author        : Pradhumnya Changdev Kalsait
# Date          : 17/01/26
# Prototype     : dict predict_heart_disease(dict)
# Input Output  : (1 input, 1 output)
#
################################################################
"""

def predict_heart_disease(input_data):
    predictor = _get_predictor()
    
    # Ensure input keys match what the model expects
    # For now, we assume input_data contains the keys:
    # 'Age', 'Gender', 'Height(cm)', 'Weight(kg)', 'Systolic_Blood_Pressure', 
    # 'Diastolic_Blood_Pressure', 'Cholesterol', 'Glucose', 'Smoking', 'Alcohol', 'Physical_Activity'
    
    result, error = predictor.predict(input_data)
    
    if error:
        return {
            "organ": "HEART",
            "disease": "Error",
            "criticality": "UNKNOWN",
            "decision": str(error)
        }
        
    return {
        "organ": "HEART",
        "disease": result['disease_type'],
        "criticality": result['risk_level'].upper(),
        "decision": f"{result['medical_action'].upper()}: {result['recommendation'].upper()}"
    }