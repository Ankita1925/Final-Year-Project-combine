# ####################################################################
# #
# # File Name :   liver_service.py
# # Description : ML prediction service for liver disease (2-stage model)
# #               Upgraded with all 10 sub-models + unit conversions
# #
# ####################################################################

# import joblib
# import numpy as np
# import logging
# import os

# logger = logging.getLogger(__name__)

# BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "ml_models")


# # -----------------------------------------------
# # Utility: A/G Ratio
# # -----------------------------------------------
# def calculate_ag_ratio(albumin: float, total_protein: float):
#     globulin = total_protein - albumin
#     if globulin <= 0:
#         logger.warning("Invalid globulin for A/G ratio")
#         return None
#     return round(albumin / globulin, 3)


# # -----------------------------------------------
# # Utility: Sub-model Confidence
# # -----------------------------------------------
# def get_submodel_probabilities(models_dict, input_array):
#     probs = {}
#     for name, model in models_dict.items():
#         try:
#             proba = model.predict_proba(input_array)[0]
#             probs[name] = round(float(max(proba)) * 100, 2)
#         except:
#             probs[name] = None
#     return probs


# # -----------------------------------------------
# # Load Models (once at startup)
# # -----------------------------------------------
# # def _load(path):
# #     full_path = os.path.join(BASE, path)
# #     return joblib.load(full_path)


# # # Primary voting classifiers
# # model1 = _load(r"D:\Be project\GUI-FYP\backend\ml_models\cirhosis\VotingClfCirh.pkl")   # 4-class: 0,1,2,3
# # model2 = _load(r"D:\Be project\GUI-FYP\backend\ml_models\yesno\VotingClfCirh.pkl")       # binary: healthy or early disease

# # # All sub-models for Model 1 (cirhosis dataset)
# # model1_sub = {
# #     "Logistic Regression":  _load("cirhosis/LogisticCirh.pkl"),
# #     "Random Forest":        _load("cirhosis/RandomForestCirh.pkl"),
# #     "AdaBoost":             _load("cirhosis/AdaboostCirh.pkl"),
# #     "KNN":                  _load("cirhosis/KNNCirh.pkl"),
# #     "Decision Tree":        _load("cirhosis/DecisionTreeCirh.pkl"),
# #     "Gradient Boosting":    _load("cirhosis/GradientBoostCirh.pkl"),
# # }

# # # All sub-models for Model 2 (ILPD dataset)
# # model2_sub = {
# #     "Logistic Regression":  _load("yesno/LogisticRegression.pkl"),
# #     "Random Forest":        _load("yesno/RandomForestClassifier.pkl"),
# #     "XGBoost":              _load("yesno/XGBoost.pkl"),
# #     "KNN":                  _load("yesno/KNN.pkl"),
# #     "Decision Tree":        _load("yesno/DecisionTreeClassifier.pkl"),
# #     "Gradient Boosting":    _load("yesno/GradientBoostingClassifier.pkl"),
# #     "AdaBoost":             _load("yesno/Adaboost.pkl"),
# # }

# def _load(path):
#     return joblib.load(path)


# ML = r"D:\Be project\GUI-FYP\backend\ml_models"

# model1 = _load(ML + r"\cirhosis\VotingClfCirh.pkl")
# model2 = _load(ML + r"\yesno\VotingClfCirh.pkl")

# model1_sub = {
#     "Logistic Regression":  _load(ML + r"\cirhosis\LogisticCirh.pkl"),
#     "Random Forest":        _load(ML + r"\cirhosis\RandomForestCirh.pkl"),
#     "AdaBoost":             _load(ML + r"\cirhosis\AdaboostCirh.pkl"),
#     "KNN":                  _load(ML + r"\cirhosis\KNNCirh.pkl"),
#     "Decision Tree":        _load(ML + r"\cirhosis\DecisionTreeCirh.pkl"),
#     "Gradient Boosting":    _load(ML + r"\cirhosis\GradientBoostCirh.pkl"),
# }

# model2_sub = {
#     "Logistic Regression":  _load(ML + r"\yesno\LogisticRegression.pkl"),
#     "Random Forest":        _load(ML + r"\yesno\RandomForestClassifier.pkl"),
#     "XGBoost":              _load(ML + r"\yesno\XGBoost.pkl"),
#     "KNN":                  _load(ML + r"\yesno\KNN.pkl"),
#     "Decision Tree":        _load(ML + r"\yesno\DecisionTreeClassifier.pkl"),
#     "Gradient Boosting":    _load(ML + r"\yesno\GradientBoostingClassifier.pkl"),
#     "AdaBoost":             _load(ML + r"\yesno\Adaboost.pkl"),
# }

# ####################################################################
# #
# # Function Name : predict_liver_disease
# # Description   : Full 2-stage ML pipeline
# # Prototype     : dict predict_liver_disease(dict)
# #
# ####################################################################
# def predict_liver_disease(input_data):

#     try:
#         # -----------------------------
#         # Extract Inputs
#         # -----------------------------
#         age        = float(input_data.get("age"))
#         gender     = int(input_data.get("gender"))
#         alb        = float(input_data.get("alb"))
#         alp        = float(input_data.get("alp"))
#         alt        = float(input_data.get("alt"))
#         ast        = float(input_data.get("ast"))
#         bil        = float(input_data.get("bil"))
#         direct_bil = float(input_data.get("direct_bilirubin"))
#         che        = float(input_data.get("che"))
#         chol       = float(input_data.get("chol"))
#         crea       = float(input_data.get("crea"))
#         ggt        = float(input_data.get("ggt"))
#         prot       = float(input_data.get("prot"))

#         inr    = input_data.get("inr")
#         sodium = input_data.get("sodium")

#         # -----------------------------
#         # Unit Conversions for Model 1
#         # (Model 1 was trained on European/SI units)
#         # ALB:  g/dL  → g/L      × 10
#         # PROT: g/dL  → g/L      × 10
#         # BIL:  mg/dL → µmol/L   × 17.1
#         # CREA: mg/dL → µmol/L   × 88.4
#         # CHOL: mg/dL → mmol/L   ÷ 38.67
#         # -----------------------------
#         alb_gL    = alb  * 10.0
#         prot_gL   = prot * 10.0
#         bil_umol  = bil  * 17.1
#         crea_umol = crea * 88.4
#         chol_mmol = chol / 38.67

#         # -----------------------------
#         # Stage 1 Prediction
#         # -----------------------------
#         m1_input = np.array([[
#             age, gender,
#             alb_gL, alp, alt, ast,
#             bil_umol, che, chol_mmol,
#             crea_umol, ggt, prot_gL
#         ]], dtype=np.float64)

#         pred1  = int(model1.predict(m1_input)[0])
#         probs1 = get_submodel_probabilities(model1_sub, m1_input)

#         # -----------------------------
#         # Initialize Response
#         # -----------------------------
#         disease     = ""
#         criticality = "LOW"
#         decision    = ""
#         organ       = "LIVER"
#         probs2      = None

#         # -----------------------------
#         # Routing Logic
#         # -----------------------------
#         if pred1 == 0:
#             # Stage 2 — Model 2 uses conventional units (no conversion needed)
#             ag = calculate_ag_ratio(alb, prot) or 0.0

#             m2_input = np.array([[
#                 age, gender,
#                 bil, direct_bil,
#                 alp, alt, ast,
#                 prot, alb, ag
#             ]], dtype=np.float64)

#             pred2  = int(model2.predict(m2_input)[0])
#             probs2 = get_submodel_probabilities(model2_sub, m2_input)

#             if pred2 == 0:
#                 disease     = "Early Liver Disease"
#                 criticality = "LOW"
#                 decision    = "LIFESTYLE CHANGES & MONITORING"
#             else:
#                 disease     = "Healthy"
#                 criticality = "NONE"
#                 decision    = "ROUTINE CHECKUP"

#         elif pred1 == 1:
#             disease     = "Hepatitis"
#             criticality = "MEDIUM"
#             decision    = "FURTHER TESTING REQUIRED"

#         elif pred1 == 2:
#             disease     = "Fibrosis"
#             criticality = "MEDIUM"
#             decision    = "SPECIALIST CONSULTATION"

#         elif pred1 == 3:
#             disease     = "Cirrhosis"
#             criticality = "HIGH"
#             if inr and bil and crea:
#                 decision = "IMMEDIATE SPECIALIST CARE + MELD SCORING"
#             else:
#                 decision = "PROVIDE INR FOR FULL SCORING"

#         else:
#             disease     = "Unknown"
#             criticality = "UNKNOWN"
#             decision    = "RETRY"

#         # -----------------------------
#         # Final Output
#         # -----------------------------
#         return {
#             "organ":              organ,
#             "disease":            disease,
#             "criticality":        criticality,
#             "decision":           decision,
#             "model1_confidence":  probs1,
#             "model2_confidence":  probs2,
#         }

#     except Exception as e:
#         logger.error("Prediction failed: %s", str(e))
#         return {
#             "organ":       "LIVER",
#             "disease":     "ERROR",
#             "criticality": "UNKNOWN",
#             "decision":    "INVALID INPUT"
#         }




####################################################################
#
# File Name   : liver_service.py
# Description : All-in-one liver disease service for GUI-FYP
#
#   Sections (in order):
#     1. ML Prediction  — 2-stage model pipeline
#     2. MELD Scoring   — MELD, MELD-Na, Child-Pugh
#     3. Report Builder — structured report dict
#     4. PDF Generator  — ReportLab PDF export (bytes)
#
#   Public API (called by liver_controller.py):
#     predict_liver_disease(input_data: dict) -> dict
#     generate_report(patient_data, prediction_result: dict) -> dict
#     generate_pdf_from_report(report: dict) -> bytes
#
####################################################################


# ====================================================================
# SECTION 1 — ML PREDICTION
# ====================================================================

import joblib
import numpy as np
import logging
import os
import pandas as pd

logger = logging.getLogger(__name__)

# -----------------------------------------------
# Model paths — relative to this file's location
# Structure: backend/ml_models/cirhosis/ and yesno/
# -----------------------------------------------
_ML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "..", "ml_models")

def _load(rel_path: str):
    """Load a .pkl model using a path relative to the ml_models directory."""
    full = os.path.normpath(os.path.join(_ML, rel_path))
    try:
        return joblib.load(full)
    except Exception as exc:
        logger.error("Failed to load model '%s': %s", full, exc)
        raise RuntimeError(f"Could not load model: {full}") from exc


# ── Primary voting classifiers ────────────────────────────────────────
model1 = _load("cirhosis/VotingClfCirh.pkl")   # 4-class: 0=Normal,1=Hepatitis,2=Fibrosis,3=Cirrhosis
model2 = _load("yesno/yesnoClfCirh.pkl")       # binary:  0=Early Liver Disease, 1=Healthy
scaler1 = _load("scalers/scaler1.pkl")
scaler2 = _load("scalers/scaler_stage2.pkl")

# ── Sub-models for Model 1 (cirhosis dataset — 4-class) ───────────────
model1_sub = {
    "Logistic Regression": _load("cirhosis/LogisticCirh.pkl"),
    "Random Forest":       _load("cirhosis/RandomForestCirh.pkl"),
    "AdaBoost":            _load("cirhosis/AdaboostCirh.pkl"),
    "KNN":                 _load("cirhosis/KNNCirh.pkl"),
    "Decision Tree":       _load("cirhosis/DecisionTreeCirh.pkl"),
    "Gradient Boosting":   _load("cirhosis/GradientBoostCirh.pkl"),
}

# ── Sub-models for Model 2 (ILPD dataset — binary) ────────────────────
model2_sub = {
    "Logistic Regression": _load("yesno/LogisticRegression.pkl"),
    "Random Forest":       _load("yesno/RandomForestClassifier.pkl"),
    "XGBoost":             _load("yesno/XGBoost.pkl"),
    "KNN":                 _load("yesno/KNN.pkl"),
    "Decision Tree":       _load("yesno/DecisionTreeClassifier.pkl"),
    "Gradient Boosting":   _load("yesno/GradientBoostingClassifier.pkl"),
    "AdaBoost":            _load("yesno/Adaboost.pkl"),
}


# -----------------------------------------------
# Utility: Albumin / Globulin ratio
# -----------------------------------------------
def _calculate_ag_ratio(albumin: float, total_protein: float) -> float | None:
    """
    Globulin = Total Protein - Albumin.
    Returns None if globulin <= 0 (invalid input).
    Normal range: 1.0 – 2.0
    """
    globulin = total_protein - albumin
    if globulin <= 0:
        logger.warning("A/G ratio skipped: globulin <= 0 (alb=%.2f, prot=%.2f)",
                       albumin, total_protein)
        return None
    return round(albumin / globulin, 3)


# -----------------------------------------------
# Utility: Sub-model confidence scores
# -----------------------------------------------
def _get_submodel_probabilities(models_dict: dict, input_array: np.ndarray) -> dict:
    """
    Collects the highest class probability from each sub-model (model's
    confidence in its own prediction). Returns None for models that don't
    support predict_proba (e.g. hard SVC).
    """
    probs = {}
    for name, model in models_dict.items():
        try:
            proba = model.predict_proba(input_array)[0]
            probs[name] = round(float(max(proba)) * 100, 2)
        except AttributeError:
            logger.warning("Model '%s' does not support predict_proba.", name)
            probs[name] = None
        except Exception as exc:
            logger.error("Error from model '%s': %s", name, exc)
            probs[name] = None
    return probs


####################################################################
#
# Function  : predict_liver_disease
# Called by : liver_controller.py → predict_liver()
# Input     : dict with keys:
#               age, gender, alb, alp, alt, ast, bil,
#               direct_bilirubin, che, chol, crea, ggt, prot
#               (optional) inr, sodium, ascites, encephalopathy
# Output    : dict — prediction result
#
# Pipeline:
#   Stage 1 — Model 1 (cirhosis dataset, 12 features, SI units)
#       0 → route to Stage 2
#       1 → Hepatitis
#       2 → Fibrosis
#       3 → Cirrhosis (MELD / Child-Pugh in report layer)
#   Stage 2 — Model 2 (ILPD dataset, 10 features, conventional units)
#       0 → Early Liver Disease
#       1 → Healthy
#
####################################################################
def predict_liver_disease(input_data: dict) -> dict:
    try:
        # ── Extract inputs ────────────────────────────────────────
        age        = float(input_data["age"])
        gender     = int(input_data["gender"])
        alb        = float(input_data["alb"])
        alp        = float(input_data["alp"])
        alt        = float(input_data["alt"])
        ast        = float(input_data["ast"])
        bil        = float(input_data["bil"])
        direct_bil = float(input_data["direct_bilirubin"])
        che        = float(input_data["che"])
        chol       = float(input_data["chol"])
        crea       = float(input_data["crea"])
        ggt        = float(input_data["ggt"])
        prot       = float(input_data["prot"])

        inr    = input_data.get("inr")
        sodium = input_data.get("sodium")

        # ── Unit conversions (MUST come before DataFrame build) ───
        alb_gL    = alb  * 10.0
        prot_gL   = prot * 10.0
        bil_umol  = bil  * 17.1
        crea_umol = crea * 88.4
        chol_mmol = chol / 38.67

        logger.debug("Unit conversions → ALB %.1f g/L | PROT %.1f g/L | "
                     "BIL %.1f µmol/L | CREA %.1f µmol/L | CHOL %.2f mmol/L",
                     alb_gL, prot_gL, bil_umol, crea_umol, chol_mmol)

        # ── Stage 1 (12 features — named DataFrame, NOT np.array) ─
        m1_input = pd.DataFrame([{
            "Age":  age,
            "Sex":  gender,
            "ALB":  alb_gL,
            "ALP":  alp,
            "ALT":  alt,
            "AST":  ast,
            "BIL":  bil_umol,
            "CHE":  che,
            "CHOL": chol_mmol,
            "CREA": crea_umol,
            "GGT":  ggt,
            "PROT": prot_gL,
        }])

        m1_scaled = scaler1.transform(m1_input)

        m1_input = pd.DataFrame(m1_scaled, columns=[
    "Age", "Sex", "ALB", "ALP", "ALT", "AST",
    "BIL", "CHE", "CHOL", "CREA", "GGT", "PROT"
])

        pred1  = int(model1.predict(m1_input)[0])
        probs1 = _get_submodel_probabilities(model1_sub, m1_input)

        # ── Base result ───────────────────────────────────────────
        result = {
            "primary_diagnosis":      None,
            "hard_voting_prediction": pred1,
            "model1_probabilities":   probs1,
            "secondary_model_used":   False,
            "model2_probabilities":   None,
            "recommendation":         None,
        }

        # ── Routing logic ─────────────────────────────────────────
        if pred1 == 0:
            result["secondary_model_used"] = True
            ag = _calculate_ag_ratio(alb, prot) or 0.0

            # Stage 2 — still np.array until you confirm model2 feature names
            m2_input = np.array([[
                age, gender,
                bil, direct_bil,
                alp, alt, ast,
                prot, alb, ag
            ]], dtype=np.float64)

            m2_input = scaler2.transform(m2_input)

            pred2  = int(model2.predict(m2_input)[0])
            probs2 = _get_submodel_probabilities(model2_sub, m2_input)
            result["model2_probabilities"] = probs2

            if pred2 == 0:
                result["primary_diagnosis"] = "Early Liver Disease"
                result["recommendation"]    = (
                    "Lifestyle modification and close monitoring advised."
                )
            else:
                result["primary_diagnosis"] = "Healthy"
                result["recommendation"]    = (
                    "Routine health check-up recommended after 6 months."
                )

        elif pred1 == 1:
            result["primary_diagnosis"] = "Hepatitis"
            result["recommendation"]    = (
                "Hepatitis viral panel (HBsAg, Anti-HCV) and full liver function tests recommended."
            )

        elif pred1 == 2:
            result["primary_diagnosis"] = "Fibrosis"
            result["recommendation"]    = (
                "Hepatologist consultation advised. FibroScan or liver biopsy may be required."
            )

        elif pred1 == 3:
            result["primary_diagnosis"] = "Cirrhosis"
            if inr and bil and crea:
                result["recommendation"] = (
                    "Advanced cirrhosis detected. "
                    "Severity scoring (MELD, Child-Pugh) included in report."
                )
            else:
                result["recommendation"] = (
                    "Cirrhosis detected. "
                    "Provide INR, Bilirubin, and Creatinine for full MELD scoring."
                )

        else:
            logger.error("Unexpected prediction value from model1: %s", pred1)
            result["primary_diagnosis"] = "Indeterminate"
            result["recommendation"]    = (
                "Unable to classify. Please review input values and retry."
            )

        return result

    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        return {
            "primary_diagnosis":      "ERROR",
            "hard_voting_prediction": None,
            "model1_probabilities":   None,
            "secondary_model_used":   False,
            "model2_probabilities":   None,
            "recommendation":         "INVALID INPUT — check all fields.",
        }

# ====================================================================
# SECTION 2 — MELD & CHILD-PUGH SCORING
# ====================================================================

import math


####################################################################
#
# Function  : calculate_meld
# Formula   : UNOS standard MELD
#               3.78·ln(Bil) + 11.2·ln(INR) + 9.57·ln(Crea) + 6.43
# MELD-Na   : applied when sodium is provided (clamped 125–137)
# Rules     : all values floored at 1.0; creatinine capped at 4.0
#
####################################################################
def calculate_meld(
    bilirubin:  float,
    inr:        float,
    creatinine: float,
    sodium:     float | None = None
) -> int | None:
    if None in (bilirubin, inr, creatinine):
        logger.warning("MELD calculation skipped: missing required values.")
        return None

    bilirubin  = max(bilirubin,  1.0)
    inr        = max(inr,        1.0)
    creatinine = min(max(creatinine, 1.0), 4.0)   # UNOS dialysis cap

    meld = (
        3.78  * math.log(bilirubin)  +
        11.2  * math.log(inr)        +
        9.57  * math.log(creatinine) +
        6.43
    )

    if sodium is not None:
        sodium = max(min(sodium, 137.0), 125.0)   # clamp per UNOS
        meld = meld + 1.32 * (137 - sodium) - (0.033 * meld * (137 - sodium))

    return round(meld)


####################################################################
#
# Function  : classify_meld
# Returns   : dict — risk_level, transplant_required,
#                    mortality_90day, description
# Threshold : MELD ≥ 15 → transplant listing (UNOS standard)
#
####################################################################
def classify_meld(meld_score: int | None) -> dict | None:
    if meld_score is None:
        return None

    if meld_score < 10:
        return {
            "risk_level":          "Low",
            "transplant_required": False,
            "mortality_90day":     "< 2%",
            "description":         "Stable cirrhosis. Regular outpatient follow-up advised.",
        }
    elif meld_score < 15:
        return {
            "risk_level":          "Moderate",
            "transplant_required": False,
            "mortality_90day":     "~6%",
            "description":         "Clinically significant liver dysfunction. Close monitoring required.",
        }
    elif meld_score < 20:
        return {
            "risk_level":          "High",
            "transplant_required": True,
            "mortality_90day":     "~20%",
            "description":         "Transplant listing evaluation recommended. Hospital admission may be needed.",
        }
    elif meld_score < 30:
        return {
            "risk_level":          "Very High",
            "transplant_required": True,
            "mortality_90day":     "~40%",
            "description":         "Urgent transplant evaluation required. High risk of decompensation.",
        }
    else:
        return {
            "risk_level":          "Critical",
            "transplant_required": True,
            "mortality_90day":     "> 70%",
            "description":         "Critical. Immediate ICU-level care and emergency transplant evaluation.",
        }


####################################################################
#
# Function  : calculate_child_pugh
# Scoring   :
#   Bilirubin (mg/dL): <2 →1 | 2-3 →2 | >3 →3
#   Albumin (g/dL):    >3.5→1 | 2.8-3.5→2 | <2.8→3
#   INR:               <1.7→1 | 1.7-2.3→2 | >2.3→3
#   Ascites (0/1/2):   None→1 | Mild→2 | Severe→3
#   Encephalopathy (0/1/2): None→1 | Gr1-2→2 | Gr3-4→3
# Classification:
#   5-6  → A (Well Compensated)
#   7-9  → B (Significant Compromise)
#   10-15→ C (Decompensated)
#
####################################################################
def calculate_child_pugh(
    bilirubin:      float,
    albumin:        float,
    inr:            float,
    ascites:        int,
    encephalopathy: int
) -> dict | None:
    if None in (bilirubin, albumin, inr, ascites, encephalopathy):
        logger.warning("Child-Pugh skipped: one or more required values are None.")
        return None

    score = 0

    score += 1 if bilirubin < 2 else (2 if bilirubin <= 3 else 3)
    score += 1 if albumin > 3.5 else (2 if albumin >= 2.8 else 3)
    score += 1 if inr < 1.7     else (2 if inr <= 2.3 else 3)
    score += (ascites + 1)
    score += (encephalopathy + 1)

    if score <= 6:
        return {
            "score":          score,
            "classification": "Child-Pugh A",
            "severity":       "Well Compensated",
            "survival_1yr":   "~100%",
            "survival_2yr":   "~85%",
            "description":    "Well-compensated cirrhosis. Medical management appropriate.",
        }
    elif score <= 9:
        return {
            "score":          score,
            "classification": "Child-Pugh B",
            "severity":       "Significant Functional Compromise",
            "survival_1yr":   "~80%",
            "survival_2yr":   "~60%",
            "description":    "Significant liver dysfunction. Specialist review and optimisation required.",
        }
    else:
        return {
            "score":          score,
            "classification": "Child-Pugh C",
            "severity":       "Decompensated",
            "survival_1yr":   "~45%",
            "survival_2yr":   "~35%",
            "description":    "Decompensated cirrhosis. Transplant evaluation strongly recommended.",
        }


# ====================================================================
# SECTION 3 — REPORT BUILDER
# ====================================================================

from datetime import datetime


# ── Lab reference ranges ─────────────────────────────────────────────
# Format: display_name → (low, high, unit)
LAB_REFERENCE_RANGES = {
    "Albumin":          (3.5,  5.0,   "g/dL"),
    "ALP":              (44,   147,   "U/L"),
    "ALT":              (7,    56,    "U/L"),
    "AST":              (10,   40,    "U/L"),
    "Total Bilirubin":  (0.2,  1.2,   "mg/dL"),
    "Direct Bilirubin": (0.0,  0.3,   "mg/dL"),
    "Cholinesterase":   (5.3,  12.9,  "kU/L"),
    "Cholesterol":      (0,    200,   "mg/dL"),
    "Creatinine":       (0.6,  1.2,   "mg/dL"),
    "GGT":              (9,    48,    "U/L"),
    "Total Protein":    (6.0,  8.3,   "g/dL"),
    "INR":              (0.8,  1.1,   "ratio"),
    "Sodium":           (136,  145,   "mEq/L"),
}

# ── Clinical interpretation notes per diagnosis ───────────────────────
CLINICAL_NOTES = {
    "Healthy": (
        "No significant liver dysfunction detected. "
        "Biochemical parameters are within acceptable limits."
    ),
    "Early Liver Disease": (
        "Early biochemical changes suggest subclinical liver dysfunction. "
        "Lifestyle modification, repeat LFTs in 3 months, and dietary review advised."
    ),
    "Hepatitis": (
        "Pattern consistent with acute or chronic hepatic inflammation. "
        "Hepatitis viral panel (HBsAg, Anti-HCV, HBeAg), autoimmune markers, "
        "and repeat LFTs are recommended."
    ),
    "Fibrosis": (
        "Biochemical pattern suggests progressive hepatic fibrosis. "
        "FibroScan (transient elastography) or liver biopsy is recommended "
        "to confirm staging. Hepatologist referral advised."
    ),
    "Cirrhosis": (
        "Findings consistent with advanced chronic liver disease (cirrhosis). "
        "Severity has been assessed using MELD and Child-Pugh scoring below. "
        "Gastroenterology / hepatology referral required."
    ),
}


def _build_lab_table(patient_data) -> list[dict]:
    """
    Builds lab rows with reference ranges + status flags.
    Accepts either a dict (from controller) or an object with attributes.
    """
    def _get(key):
        if isinstance(patient_data, dict):
            return patient_data.get(key)
        return getattr(patient_data, key, None)

    raw = {
        "Albumin":          _get("alb"),
        "ALP":              _get("alp"),
        "ALT":              _get("alt"),
        "AST":              _get("ast"),
        "Total Bilirubin":  _get("bil"),
        "Direct Bilirubin": _get("direct_bilirubin"),
        "Cholinesterase":   _get("che"),
        "Cholesterol":      _get("chol"),
        "Creatinine":       _get("crea"),
        "GGT":              _get("ggt"),
        "Total Protein":    _get("prot"),
        "INR":              _get("inr"),
        "Sodium":           _get("sodium"),
    }

    table = []
    for name, value in raw.items():
        low, high, unit = LAB_REFERENCE_RANGES[name]
        if value is None:
            status = "Not Provided"
        elif value < low:
            status = "Low"
        elif value > high:
            status = "High"
        else:
            status = "Normal"

        table.append({
            "name":   name,
            "value":  value,
            "unit":   unit,
            "low":    low,
            "high":   high,
            "status": status,
        })
    return table


def _build_model_confidence(probabilities: dict | None) -> dict:
    """
    Summarises submodel confidence scores.
    Returns scores dict, average %, and agreement level.
    """
    if not probabilities:
        return {"scores": {}, "average": None, "agreement": "Not Available"}

    valid = {k: v for k, v in probabilities.items() if isinstance(v, (int, float))}
    if not valid:
        return {"scores": {}, "average": None, "agreement": "Not Available"}

    average = round(sum(valid.values()) / len(valid), 1)
    agreement = "High" if average >= 85 else ("Moderate" if average >= 70 else "Low")

    return {"scores": valid, "average": average, "agreement": agreement}


####################################################################
#
# Function  : generate_report
# Called by : liver_controller.py (or wherever report is needed)
# Input     : patient_data  — dict or object with lab field attributes
#             prediction_result — dict returned by predict_liver_disease()
# Output    : fully structured report dict consumed by:
#               - report.html  (Jinja2 template)
#               - generate_pdf_from_report() below
#
####################################################################
def generate_report(patient_data, prediction_result: dict) -> dict:
    def _get(key):
        if isinstance(patient_data, dict):
            return patient_data.get(key)
        return getattr(patient_data, key, None)

    disease          = prediction_result.get("primary_diagnosis")
    prediction_class = prediction_result.get("hard_voting_prediction")
    recommendation   = prediction_result.get("recommendation")
    secondary_used   = prediction_result.get("secondary_model_used", False)

    model1_probs = prediction_result.get("model1_probabilities")
    model2_probs = prediction_result.get("model2_probabilities")
    active_probs = model2_probs if secondary_used else model1_probs

    report = {

        "report_metadata": {
            "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "system":       "OnlyLiver AI Decision Support System v2.0",
        },

        "patient_information": {
            "age":    _get("age"),
            "gender": "Male" if _get("gender") == 1 else "Female",
        },

        "laboratory_values": _build_lab_table(patient_data),

        "ai_diagnosis": {
            "primary_diagnosis":    disease,
            "prediction_class":     prediction_class,
            "secondary_model_used": secondary_used,
            "pipeline_summary": (
                "Stage 1 (Cirrhosis model) → Stage 2 (ILPD early assessment model)"
                if secondary_used else
                "Stage 1 (Cirrhosis model) — final diagnosis"
            ),
        },

        "model_confidence": {
            "model1_submodels": _build_model_confidence(model1_probs),
            "model2_submodels": _build_model_confidence(model2_probs),
            "active":           _build_model_confidence(active_probs),
        },

        "clinical_interpretation": CLINICAL_NOTES.get(
            disease, "Further clinical evaluation is required."
        ),

        "severity_assessment": None,   # populated below for Cirrhosis only

        "final_recommendation": recommendation,

        "medical_disclaimer": (
            "This AI-generated report is a clinical decision support tool only. "
            "It does not replace the judgement of a licensed medical professional. "
            "All findings must be interpreted in the context of the full clinical picture."
        ),
    }

    # ── Severity Assessment — Cirrhosis only (prediction_class == 3) ──
    if prediction_class == 3:

        meld_score = calculate_meld(
            _get("bil"),
            _get("inr"),
            _get("crea"),
            _get("sodium"),
        )
        meld_result = classify_meld(meld_score)

        child_pugh = calculate_child_pugh(
            _get("bil"),
            _get("alb"),
            _get("inr"),
            _get("ascites"),
            _get("encephalopathy"),
        )

        report["severity_assessment"] = {
            "meld_score":          meld_score,
            "meld":                meld_result,
            "child_pugh":          child_pugh,
            "transplant_required": (
                meld_result["transplant_required"] if meld_result else False
            ),
        }

    return report


# ====================================================================
# SECTION 4 — PDF GENERATOR
# ====================================================================

import io

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)


# ── Colour palette ────────────────────────────────────────────────────
_NAVY       = colors.HexColor("#1B3A5C")
_TEAL       = colors.HexColor("#1A7A8A")
_LIGHT_TEAL = colors.HexColor("#E8F4F6")
_RED        = colors.HexColor("#C0392B")
_AMBER      = colors.HexColor("#E67E22")
_GREEN      = colors.HexColor("#27AE60")
_LIGHT_GREY = colors.HexColor("#F5F5F5")
_MID_GREY   = colors.HexColor("#CCCCCC")
_DARK_GREY  = colors.HexColor("#555555")
_WHITE      = colors.white
_BLACK      = colors.black

_STATUS_COLORS = {
    "Normal":       _GREEN,
    "High":         _RED,
    "Low":          _AMBER,
    "Not Provided": _MID_GREY,
}


def _build_pdf_styles() -> dict:
    styles = {
        "title": ParagraphStyle(
            "ReportTitle", fontSize=22, fontName="Helvetica-Bold",
            textColor=_WHITE, alignment=TA_CENTER, spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle", fontSize=10, fontName="Helvetica",
            textColor=colors.HexColor("#CCE8ED"), alignment=TA_CENTER, spaceAfter=0,
        ),
        "section_heading": ParagraphStyle(
            "SectionHeading", fontSize=12, fontName="Helvetica-Bold",
            textColor=_NAVY, spaceBefore=10, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body", fontSize=9, fontName="Helvetica",
            textColor=_DARK_GREY, leading=14, spaceAfter=4,
        ),
        "body_bold": ParagraphStyle(
            "BodyBold", fontSize=9, fontName="Helvetica-Bold",
            textColor=_BLACK, leading=14,
        ),
        "small": ParagraphStyle(
            "Small", fontSize=8, fontName="Helvetica",
            textColor=_DARK_GREY, leading=12,
        ),
        "disclaimer": ParagraphStyle(
            "Disclaimer", fontSize=8, fontName="Helvetica-Oblique",
            textColor=_DARK_GREY, leading=11,
        ),
    }
    return styles


# ── PDF internal helpers ──────────────────────────────────────────────

def _section_heading(text, styles):
    return KeepTogether([
        Paragraph(text.upper(), styles["section_heading"]),
        HRFlowable(width="100%", thickness=1, color=_TEAL, spaceAfter=6),
    ])


def _two_col_table(l_label, l_val, r_label, r_val, styles):
    data = [[
        Paragraph(f"<b>{l_label}</b>", styles["body"]),
        Paragraph(str(l_val),          styles["body"]),
        Paragraph(f"<b>{r_label}</b>", styles["body"]),
        Paragraph(str(r_val),          styles["body"]),
    ]]
    t = Table(data, colWidths=[38*mm, 52*mm, 38*mm, 52*mm])
    t.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _info_box(text, bg_color, text_color, styles):
    data = [[Paragraph(text, ParagraphStyle(
        "InfoBox", fontSize=10, fontName="Helvetica-Bold",
        textColor=text_color, alignment=TA_CENTER,
    ))]]
    t = Table(data, colWidths=[180*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), bg_color),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def _pdf_build_header(report, styles, page_width):
    meta = report.get("report_metadata", {})
    data = [
        [Paragraph("OnlyLiver AI", styles["title"])],
        [Paragraph("Clinical Liver Disease Decision Support Report", styles["subtitle"])],
        [Paragraph(
            f"Generated: {meta.get('generated_at', '')}  |  {meta.get('system', '')}",
            styles["subtitle"]
        )],
    ]
    t = Table(data, colWidths=[page_width])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), _NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
    ]))
    return t


def _pdf_patient_section(report, styles):
    story = [_section_heading("Patient Information", styles)]
    info  = report.get("patient_information", {})
    story.append(_two_col_table("Age", info.get("age", "—"),
                                "Gender", info.get("gender", "—"), styles))
    story.append(Spacer(1, 4*mm))
    return story


def _pdf_diagnosis_section(report, styles):
    story = [_section_heading("AI Diagnosis", styles)]
    diag  = report.get("ai_diagnosis", {})

    story.append(_info_box(
        f"Diagnosis: {diag.get('primary_diagnosis', 'Unknown')}",
        _TEAL, _WHITE, styles,
    ))
    story.append(Spacer(1, 3*mm))
    story.append(_two_col_table(
        "Prediction Class", diag.get("prediction_class", "—"),
        "Secondary Model Used", "Yes" if diag.get("secondary_model_used") else "No",
        styles,
    ))
    story.append(Paragraph(f"<b>Pipeline:</b> {diag.get('pipeline_summary', '—')}", styles["body"]))
    story.append(Spacer(1, 4*mm))
    return story


def _pdf_lab_section(report, styles):
    story    = [_section_heading("Laboratory Values", styles)]
    lab_rows = report.get("laboratory_values", [])

    header = [
        Paragraph("<b>Parameter</b>",      styles["body_bold"]),
        Paragraph("<b>Value</b>",           styles["body_bold"]),
        Paragraph("<b>Unit</b>",            styles["body_bold"]),
        Paragraph("<b>Reference Range</b>", styles["body_bold"]),
        Paragraph("<b>Status</b>",          styles["body_bold"]),
    ]
    rows       = [header]
    row_styles = []

    for i, lab in enumerate(lab_rows, start=1):
        value  = lab.get("value")
        status = lab.get("status", "")
        s_color = _STATUS_COLORS.get(status, _MID_GREY)

        rows.append([
            Paragraph(lab.get("name", ""), styles["body"]),
            Paragraph(f"{value}" if value is not None else "—", styles["body"]),
            Paragraph(lab.get("unit", ""), styles["body"]),
            Paragraph(f"{lab.get('low', '')} – {lab.get('high', '')}", styles["body"]),
            Paragraph(f"<b>{status}</b>", ParagraphStyle(
                "StatusCell", fontSize=8, fontName="Helvetica-Bold",
                textColor=s_color, alignment=TA_CENTER,
            )),
        ])
        row_styles.append(("BACKGROUND", (0, i), (-1, i),
                           _LIGHT_GREY if i % 2 == 0 else _WHITE))

    t = Table(rows, colWidths=[52*mm, 28*mm, 22*mm, 45*mm, 33*mm], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), _NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), _WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",          (0, 0), (-1, -1), 0.5, _MID_GREY),
        *row_styles,
    ]))
    story.append(t)
    story.append(Spacer(1, 4*mm))
    return story


def _pdf_confidence_section(report, styles):
    story  = [_section_heading("Model Confidence", styles)]
    conf   = report.get("model_confidence", {})
    active = conf.get("active", {})

    story.append(Paragraph(
        f"<b>Active Model Group:</b>  "
        f"Average Confidence: <b>{active.get('average', '—')}%</b>  |  "
        f"Agreement: <b>{active.get('agreement', '—')}</b>",
        styles["body"]
    ))
    story.append(Spacer(1, 2*mm))

    def _conf_table(label, conf_dict):
        scores = conf_dict.get("scores", {})
        if not scores:
            return []
        rows = [[
            Paragraph(f"<b>{label}</b>", styles["body_bold"]),
            Paragraph("<b>Confidence</b>", styles["body_bold"]),
        ]]
        for name, pct in scores.items():
            rows.append([Paragraph(name, styles["body"]),
                         Paragraph(f"{pct}%", styles["body"])])
        t = Table(rows, colWidths=[90*mm, 35*mm])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), _LIGHT_TEAL),
            ("GRID",          (0, 0), (-1, -1), 0.5, _MID_GREY),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ]))
        return [t, Spacer(1, 2*mm)]

    story += _conf_table("Cirrhosis Model — Submodels", conf.get("model1_submodels", {}))
    m2 = conf.get("model2_submodels", {})
    if m2.get("scores"):
        story += _conf_table("ILPD Early Assessment — Submodels", m2)

    story.append(Spacer(1, 4*mm))
    return story


def _pdf_clinical_section(report, styles):
    story = [_section_heading("Clinical Interpretation", styles)]
    story.append(Paragraph(
        report.get("clinical_interpretation", "No interpretation available."),
        styles["body"]
    ))
    story.append(Spacer(1, 4*mm))
    return story


def _pdf_severity_section(report, styles):
    severity = report.get("severity_assessment")
    if not severity:
        return []

    story      = [_section_heading("Severity Assessment (Cirrhosis)", styles)]
    meld_score = severity.get("meld_score")
    meld       = severity.get("meld") or {}
    child_pugh = severity.get("child_pugh") or {}
    transplant = severity.get("transplant_required", False)

    story.append(_info_box(
        "TRANSPLANT EVALUATION RECOMMENDED" if transplant else "Medical Management Advised",
        _RED if transplant else _GREEN, _WHITE, styles,
    ))
    story.append(Spacer(1, 3*mm))

    # MELD table
    if meld_score is not None:
        story.append(Paragraph("<b>MELD Score</b>", styles["body_bold"]))
        t = Table([
            ["MELD Score",       str(meld_score)],
            ["Risk Level",       meld.get("risk_level", "—")],
            ["90-Day Mortality", meld.get("mortality_90day", "—")],
            ["Description",      meld.get("description", "—")],
        ], colWidths=[55*mm, 125*mm])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (0, -1), _LIGHT_TEAL),
            ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
            ("GRID",          (0, 0), (-1, -1), 0.5, _MID_GREY),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ]))
        story.append(t)
        story.append(Spacer(1, 3*mm))
    else:
        story.append(Paragraph(
            "MELD score unavailable — INR, Bilirubin, and Creatinine are required.",
            styles["body"]
        ))

    # Child-Pugh table
    if child_pugh:
        story.append(Paragraph("<b>Child-Pugh Score</b>", styles["body_bold"]))
        t = Table([
            ["Score",           str(child_pugh.get("score", "—"))],
            ["Classification",  child_pugh.get("classification", "—")],
            ["Severity",        child_pugh.get("severity", "—")],
            ["1-Year Survival", child_pugh.get("survival_1yr", "—")],
            ["2-Year Survival", child_pugh.get("survival_2yr", "—")],
            ["Description",     child_pugh.get("description", "—")],
        ], colWidths=[55*mm, 125*mm])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (0, -1), _LIGHT_TEAL),
            ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
            ("GRID",          (0, 0), (-1, -1), 0.5, _MID_GREY),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ]))
        story.append(t)
    else:
        story.append(Paragraph(
            "Child-Pugh score unavailable — Ascites and Encephalopathy grades are required.",
            styles["body"]
        ))

    story.append(Spacer(1, 4*mm))
    return story


def _pdf_recommendation_section(report, styles):
    story = [_section_heading("Final Recommendation", styles)]
    story.append(Paragraph(
        report.get("final_recommendation", "No recommendation available."),
        styles["body"]
    ))
    story.append(Spacer(1, 4*mm))
    return story


def _pdf_disclaimer(report, styles):
    return [
        HRFlowable(width="100%", thickness=1, color=_MID_GREY, spaceBefore=4),
        Spacer(1, 2*mm),
        Paragraph(
            f"<i>{report.get('medical_disclaimer', '')}</i>",
            styles["disclaimer"]
        ),
    ]


####################################################################
#
# Function  : generate_pdf_from_report
# Called by : liver_controller.py → download_report() endpoint
# Input     : report dict from generate_report()
# Output    : PDF as raw bytes (ready to stream as application/pdf)
#
####################################################################
def generate_pdf_from_report(report: dict) -> bytes:
    buffer = io.BytesIO()
    margin = 15 * mm

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=margin,  bottomMargin=margin,
        title="OnlyLiver AI Clinical Report",
        author="OnlyLiver AI Decision Support System",
    )

    page_width = A4[0] - 2 * margin
    styles     = _build_pdf_styles()
    story      = []

    story.append(_pdf_build_header(report, styles, page_width))
    story.append(Spacer(1, 5*mm))
    story += _pdf_patient_section(report, styles)
    story += _pdf_diagnosis_section(report, styles)
    story += _pdf_lab_section(report, styles)
    story += _pdf_confidence_section(report, styles)
    story += _pdf_clinical_section(report, styles)
    story += _pdf_severity_section(report, styles)     # renders only for Cirrhosis
    story += _pdf_recommendation_section(report, styles)
    story += _pdf_disclaimer(report, styles)

    doc.build(story)
    return buffer.getvalue()