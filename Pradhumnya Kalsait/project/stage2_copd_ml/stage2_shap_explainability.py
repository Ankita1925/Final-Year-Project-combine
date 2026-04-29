import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# Load trained Stage-2 model
# =====================================================
pipeline = joblib.load("models_stage2/GradientBoosting.pkl")

scaler = pipeline.named_steps["scaler"]
model = pipeline.named_steps["model"]

# =====================================================
# Load dataset
# =====================================================
df = pd.read_csv("data/copd_clinical.csv")

binary_map = {
    "Yes": 1, "Non": 0,
    "Normal": 0, "Higher": 1,
    "Low": 0, "High": 1,
    "Purulent": 1
}

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].map(binary_map).fillna(0)

# Severity label (same logic)
def fev1_to_severity(fev1):
    if fev1 >= 80:
        return 0
    elif fev1 >= 50:
        return 1
    elif fev1 >= 30:
        return 2
    else:
        return 3

df["Severity"] = df["FEV1"].apply(fev1_to_severity)

X = df.drop(["FEV1", "Severity"], axis=1)
X_scaled = scaler.transform(X)

feature_names = X.columns.tolist()

# =====================================================
# SHAP Explainer
# =====================================================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled)

# =====================================================
# Global Feature Importance
# =====================================================
shap.summary_plot(
    shap_values,
    X_scaled,
    feature_names=feature_names,
    show=False
)
plt.title("SHAP Summary Plot – COPD Severity Prediction")
plt.tight_layout()
plt.savefig("plots_stage2/shap_summary.png", dpi=300)
plt.show()

# =====================================================
# Bar Plot (Mean |SHAP|)
# =====================================================
shap.summary_plot(
    shap_values,
    X_scaled,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)
plt.title("SHAP Feature Importance (Mean Impact)")
plt.tight_layout()
plt.savefig("plots_stage2/shap_bar.png", dpi=300)
plt.show()

# =====================================================
# Individual Patient Explanation (Optional)
# =====================================================
patient_id = 0  # example
shap.force_plot(
    explainer.expected_value[3],
    shap_values[3][patient_id],
    X.iloc[patient_id],
    matplotlib=True
)
plt.savefig("plots_stage2/shap_force_patient.png", dpi=300)
