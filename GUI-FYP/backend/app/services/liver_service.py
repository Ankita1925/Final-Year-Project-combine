####################################################################
#
# File Name :   liver_service.py
# Description : ML prediction service for liver disease (2-stage model)
#               Upgraded with all 10 sub-models + unit conversions
#
####################################################################

import joblib
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "ml_models")


# -----------------------------------------------
# Utility: A/G Ratio
# -----------------------------------------------
def calculate_ag_ratio(albumin: float, total_protein: float):
    globulin = total_protein - albumin
    if globulin <= 0:
        logger.warning("Invalid globulin for A/G ratio")
        return None
    return round(albumin / globulin, 3)


# -----------------------------------------------
# Utility: Sub-model Confidence
# -----------------------------------------------
def get_submodel_probabilities(models_dict, input_array):
    probs = {}
    for name, model in models_dict.items():
        try:
            proba = model.predict_proba(input_array)[0]
            probs[name] = round(float(max(proba)) * 100, 2)
        except:
            probs[name] = None
    return probs


# -----------------------------------------------
# Load Models (once at startup)
# -----------------------------------------------
# def _load(path):
#     full_path = os.path.join(BASE, path)
#     return joblib.load(full_path)


# # Primary voting classifiers
# model1 = _load(r"D:\Be project\GUI-FYP\backend\ml_models\cirhosis\VotingClfCirh.pkl")   # 4-class: 0,1,2,3
# model2 = _load(r"D:\Be project\GUI-FYP\backend\ml_models\yesno\VotingClfCirh.pkl")       # binary: healthy or early disease

# # All sub-models for Model 1 (cirhosis dataset)
# model1_sub = {
#     "Logistic Regression":  _load("cirhosis/LogisticCirh.pkl"),
#     "Random Forest":        _load("cirhosis/RandomForestCirh.pkl"),
#     "AdaBoost":             _load("cirhosis/AdaboostCirh.pkl"),
#     "KNN":                  _load("cirhosis/KNNCirh.pkl"),
#     "Decision Tree":        _load("cirhosis/DecisionTreeCirh.pkl"),
#     "Gradient Boosting":    _load("cirhosis/GradientBoostCirh.pkl"),
# }

# # All sub-models for Model 2 (ILPD dataset)
# model2_sub = {
#     "Logistic Regression":  _load("yesno/LogisticRegression.pkl"),
#     "Random Forest":        _load("yesno/RandomForestClassifier.pkl"),
#     "XGBoost":              _load("yesno/XGBoost.pkl"),
#     "KNN":                  _load("yesno/KNN.pkl"),
#     "Decision Tree":        _load("yesno/DecisionTreeClassifier.pkl"),
#     "Gradient Boosting":    _load("yesno/GradientBoostingClassifier.pkl"),
#     "AdaBoost":             _load("yesno/Adaboost.pkl"),
# }

def _load(path):
    return joblib.load(path)


ML = r"D:\Be project\GUI-FYP\backend\ml_models"

model1 = _load(ML + r"\cirhosis\VotingClfCirh.pkl")
model2 = _load(ML + r"\yesno\VotingClfCirh.pkl")

model1_sub = {
    "Logistic Regression":  _load(ML + r"\cirhosis\LogisticCirh.pkl"),
    "Random Forest":        _load(ML + r"\cirhosis\RandomForestCirh.pkl"),
    "AdaBoost":             _load(ML + r"\cirhosis\AdaboostCirh.pkl"),
    "KNN":                  _load(ML + r"\cirhosis\KNNCirh.pkl"),
    "Decision Tree":        _load(ML + r"\cirhosis\DecisionTreeCirh.pkl"),
    "Gradient Boosting":    _load(ML + r"\cirhosis\GradientBoostCirh.pkl"),
}

model2_sub = {
    "Logistic Regression":  _load(ML + r"\yesno\LogisticRegression.pkl"),
    "Random Forest":        _load(ML + r"\yesno\RandomForestClassifier.pkl"),
    "XGBoost":              _load(ML + r"\yesno\XGBoost.pkl"),
    "KNN":                  _load(ML + r"\yesno\KNN.pkl"),
    "Decision Tree":        _load(ML + r"\yesno\DecisionTreeClassifier.pkl"),
    "Gradient Boosting":    _load(ML + r"\yesno\GradientBoostingClassifier.pkl"),
    "AdaBoost":             _load(ML + r"\yesno\Adaboost.pkl"),
}

####################################################################
#
# Function Name : predict_liver_disease
# Description   : Full 2-stage ML pipeline
# Prototype     : dict predict_liver_disease(dict)
#
####################################################################
def predict_liver_disease(input_data):

    try:
        # -----------------------------
        # Extract Inputs
        # -----------------------------
        age        = float(input_data.get("age"))
        gender     = int(input_data.get("gender"))
        alb        = float(input_data.get("alb"))
        alp        = float(input_data.get("alp"))
        alt        = float(input_data.get("alt"))
        ast        = float(input_data.get("ast"))
        bil        = float(input_data.get("bil"))
        direct_bil = float(input_data.get("direct_bilirubin"))
        che        = float(input_data.get("che"))
        chol       = float(input_data.get("chol"))
        crea       = float(input_data.get("crea"))
        ggt        = float(input_data.get("ggt"))
        prot       = float(input_data.get("prot"))

        inr    = input_data.get("inr")
        sodium = input_data.get("sodium")

        # -----------------------------
        # Unit Conversions for Model 1
        # (Model 1 was trained on European/SI units)
        # ALB:  g/dL  → g/L      × 10
        # PROT: g/dL  → g/L      × 10
        # BIL:  mg/dL → µmol/L   × 17.1
        # CREA: mg/dL → µmol/L   × 88.4
        # CHOL: mg/dL → mmol/L   ÷ 38.67
        # -----------------------------
        alb_gL    = alb  * 10.0
        prot_gL   = prot * 10.0
        bil_umol  = bil  * 17.1
        crea_umol = crea * 88.4
        chol_mmol = chol / 38.67

        # -----------------------------
        # Stage 1 Prediction
        # -----------------------------
        m1_input = np.array([[
            age, gender,
            alb_gL, alp, alt, ast,
            bil_umol, che, chol_mmol,
            crea_umol, ggt, prot_gL
        ]], dtype=np.float64)

        pred1  = int(model1.predict(m1_input)[0])
        probs1 = get_submodel_probabilities(model1_sub, m1_input)

        # -----------------------------
        # Initialize Response
        # -----------------------------
        disease     = ""
        criticality = "LOW"
        decision    = ""
        organ       = "LIVER"
        probs2      = None

        # -----------------------------
        # Routing Logic
        # -----------------------------
        if pred1 == 0:
            # Stage 2 — Model 2 uses conventional units (no conversion needed)
            ag = calculate_ag_ratio(alb, prot) or 0.0

            m2_input = np.array([[
                age, gender,
                bil, direct_bil,
                alp, alt, ast,
                prot, alb, ag
            ]], dtype=np.float64)

            pred2  = int(model2.predict(m2_input)[0])
            probs2 = get_submodel_probabilities(model2_sub, m2_input)

            if pred2 == 0:
                disease     = "Early Liver Disease"
                criticality = "LOW"
                decision    = "LIFESTYLE CHANGES & MONITORING"
            else:
                disease     = "Healthy"
                criticality = "NONE"
                decision    = "ROUTINE CHECKUP"

        elif pred1 == 1:
            disease     = "Hepatitis"
            criticality = "MEDIUM"
            decision    = "FURTHER TESTING REQUIRED"

        elif pred1 == 2:
            disease     = "Fibrosis"
            criticality = "MEDIUM"
            decision    = "SPECIALIST CONSULTATION"

        elif pred1 == 3:
            disease     = "Cirrhosis"
            criticality = "HIGH"
            if inr and bil and crea:
                decision = "IMMEDIATE SPECIALIST CARE + MELD SCORING"
            else:
                decision = "PROVIDE INR FOR FULL SCORING"

        else:
            disease     = "Unknown"
            criticality = "UNKNOWN"
            decision    = "RETRY"

        # -----------------------------
        # Final Output
        # -----------------------------
        return {
            "organ":              organ,
            "disease":            disease,
            "criticality":        criticality,
            "decision":           decision,
            "model1_confidence":  probs1,
            "model2_confidence":  probs2,
        }

    except Exception as e:
        logger.error("Prediction failed: %s", str(e))
        return {
            "organ":       "LIVER",
            "disease":     "ERROR",
            "criticality": "UNKNOWN",
            "decision":    "INVALID INPUT"
        }