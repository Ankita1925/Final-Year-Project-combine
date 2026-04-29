# =====================================================
# STAGE-2 COPD SEVERITY (WITHOUT FEV1)
# Real AI Severity Model
# =====================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    VotingClassifier,
    StackingClassifier
)

# =====================================================
# CONFIG
# =====================================================
DATA_FILE = "PatientCategorical.csv"
TARGET = "COPD GOLD"

MODEL_DIR = "models_stage2"
PLOT_DIR = "plots_stage2"
RISK_DIR = "risk_stage2"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RISK_DIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(DATA_FILE)

if TARGET not in df.columns:
    raise ValueError("Target column missing!")

# 🔥 REMOVE FEV1 (Critical)
X_raw = df.drop([TARGET, "FEV1"], axis=1)

y = df[TARGET]

feature_names = X_raw.columns

print("\n[INFO] Dataset Loaded:", df.shape)
print("[INFO] Features Used:", list(feature_names))
print("\n[INFO] GOLD Distribution:\n")
print(y.value_counts())

# =====================================================
# CROSS-VALIDATION FUNCTION (NO LEAKAGE)
# =====================================================
def cross_validate_model(model, X_raw, y, folds=5):

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    acc, prec, rec, f1 = [], [], [], []

    for train_idx, test_idx in skf.split(X_raw, y):

        X_tr_raw = X_raw.iloc[train_idx]
        X_te_raw = X_raw.iloc[test_idx]
        y_tr = y.iloc[train_idx]
        y_te = y.iloc[test_idx]

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", clone(model))
        ])

        pipeline.fit(X_tr_raw, y_tr)
        y_pred = pipeline.predict(X_te_raw)

        acc.append(accuracy_score(y_te, y_pred))
        prec.append(precision_score(y_te, y_pred, average="weighted"))
        rec.append(recall_score(y_te, y_pred, average="weighted"))
        f1.append(f1_score(y_te, y_pred, average="weighted"))

    return {
        "Accuracy": np.mean(acc),
        "Precision": np.mean(prec),
        "Recall": np.mean(rec),
        "F1": np.mean(f1)
    }

# =====================================================
# MODELS (14 TOTAL)
# =====================================================
models = {

    "LogisticRegression": LogisticRegression(max_iter=5000),
    "LinearSVM": LinearSVC(),
    "RBFSVM": SVC(kernel="rbf", probability=True),

    "DecisionTree": DecisionTreeClassifier(random_state=42),

    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=300, random_state=42),
    "BaggingDT": BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=100,
        random_state=42
    ),

    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),

    "GaussianNB": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=7)
}

# Voting
models["VotingHard"] = VotingClassifier(
    estimators=[
        ("lr", models["LogisticRegression"]),
        ("rf", models["RandomForest"]),
        ("dt", models["DecisionTree"])
    ],
    voting="hard"
)

models["VotingSoft"] = VotingClassifier(
    estimators=[
        ("lr", models["LogisticRegression"]),
        ("rf", models["RandomForest"]),
        ("gb", models["GradientBoosting"])
    ],
    voting="soft"
)

# Stacking
models["Stacking"] = StackingClassifier(
    estimators=[
        ("rf", models["RandomForest"]),
        ("gb", models["GradientBoosting"]),
        ("svm", models["RBFSVM"])
    ],
    final_estimator=LogisticRegression(max_iter=3000)
)

# =====================================================
# TRAIN + EVALUATE
# =====================================================
results = []

for name, model in models.items():

    metrics = cross_validate_model(model, X_raw, y)
    metrics["Model"] = name
    results.append(metrics)

    # Train final model on full data
    final_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", clone(model))
    ])

    final_pipeline.fit(X_raw, y)
    joblib.dump(final_pipeline, f"{MODEL_DIR}/{name}.pkl")

    print(f"[OK] {name} trained & saved")

df_metrics = pd.DataFrame(results)[
    ["Model", "Accuracy", "Precision", "Recall", "F1"]
]

df_metrics = df_metrics.sort_values(by="Accuracy", ascending=False)

print("\n================ STAGE-2 MODEL COMPARISON ================\n")
print(df_metrics.to_string(index=False))
print("\n===========================================================\n")

# =====================================================
# VISUALIZATION
# =====================================================
plot_df = df_metrics.melt(
    id_vars="Model",
    value_vars=["Accuracy", "Recall", "F1"],
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(14, 7))
sns.barplot(
    data=plot_df,
    x="Model",
    y="Score",
    hue="Metric",
    palette="viridis"
)

plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.title("Stage-2 COPD Severity (Without FEV1)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/stage2_model_comparison.png")
plt.show()

# =====================================================
# RISK ANALYSIS (TOP-2 MODELS)
# =====================================================
top_models = df_metrics.head(2)

print("\n[INFO] Top-2 Models for Risk Analysis:\n")
print(top_models)

for _, row in top_models.iterrows():

    model_name = row["Model"]
    pipeline = joblib.load(f"{MODEL_DIR}/{model_name}.pkl")
    model = pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        label = "Importance"
    else:
        importance = np.abs(model.coef_).mean(axis=0)
        label = "Risk Weight"

    df_risk = pd.DataFrame({
        "Feature": feature_names,
        label: importance
    }).sort_values(by=label, ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(
        x=label,
        y="Feature",
        data=df_risk,
        palette="magma"
    )

    plt.title(f"Feature Risk Analysis – {model_name}")
    plt.tight_layout()
    plt.savefig(f"{RISK_DIR}/{model_name}_risk.png")
    plt.show()