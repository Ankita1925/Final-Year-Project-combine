import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC


from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)

from utils import load_txt_file


import importlib
import utils
importlib.reload(utils)
from utils import load_txt_file


# =====================================================
# Cross-Validation Function
# =====================================================
def cross_validate_model(model, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    acc, rec, f1 = [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = clone(model)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc.append(accuracy_score(y_test, y_pred))
        rec.append(recall_score(y_test, y_pred, average="weighted"))
        f1.append(f1_score(y_test, y_pred, average="weighted"))

    return {
        "Accuracy_mean": np.mean(acc),
        "Accuracy_std": np.std(acc),
        "Recall_mean": np.mean(rec),
        "Recall_std": np.std(rec),
        "F1_mean": np.mean(f1),
        "F1_std": np.std(f1)
    }


# =====================================================
# Load Dataset (AIR REMOVED)
# =====================================================
data_paths = {
    "COPD": "data/COPD.csv",
    "SMOKERS": "data/SMOKERS.csv",
    "CONTROL": "data/CONTROL.csv"
}

X_list, y_list = [], []

print("\n[DEBUG] Loading datasets...\n")

for label, path in data_paths.items():
    X, y = load_txt_file(path, label)

    print(f"[DEBUG] {label}")
    print(f"        X shape: {X.shape}")
    print(f"        y shape: {y.shape}")
    print("-" * 40)

    if X_list and X.shape[1] != X_list[0].shape[1]:
        raise ValueError("Feature dimension mismatch detected!")

    X_list.append(X)
    y_list.append(y)

X = np.vstack(X_list)
y = np.hstack(y_list)

print(f"\n[DEBUG] Final dataset: X={X.shape}, y={y.shape}")


# =====================================================
# Models
# =====================================================
# =====================================================
# Models (14 Total)
# =====================================================
models = {
    # Linear / Margin
    "LogisticRegression": LogisticRegression(max_iter=5000),
    "LinearSVM": LinearSVC(),
    "RBFSVM": SVC(kernel="rbf", probability=True),

    # Tree
    "DecisionTree": DecisionTreeClassifier(random_state=42),

    # Bagging / Forest
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=200, random_state=42),
    "BaggingDT": BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=50,
        random_state=42
    ),

    # Boosting
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),

    # Probabilistic
    "GaussianNB": GaussianNB(),

    # Neighbors
    "KNN": KNeighborsClassifier(n_neighbors=7)
}

# =====================================================
# Voting Ensembles
# =====================================================
models["VotingHard"] = VotingClassifier(
    estimators=[
        ("lr", models["LogisticRegression"]),
        ("dt", models["DecisionTree"]),
        ("rf", models["RandomForest"])
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

# =====================================================
# Stacking
# =====================================================
models["Stacking"] = StackingClassifier(
    estimators=[
        ("dt", models["DecisionTree"]),
        ("rf", models["RandomForest"]),
        ("svm", models["RBFSVM"])
    ],
    final_estimator=LogisticRegression()
)


# =====================================================
# Training + Stratified K-Fold Evaluation
# =====================================================
results = []
os.makedirs("models/saved_models", exist_ok=True)

for name, model in models.items():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    cv_metrics = cross_validate_model(pipeline, X, y)
    cv_metrics["Model"] = name
    results.append(cv_metrics)

    pipeline.fit(X, y)
    joblib.dump(pipeline, f"models/saved_models/{name}.pkl")

    print(f"[INFO] {name} CV done & model saved")
    df_results = pd.DataFrame(results)
    print("\n[RESULTS]")

# =====================================================
# Terminal Performance Table
# =====================================================

df_results = pd.DataFrame(results)

performance_table = df_results[[
    "Model",
    "Accuracy_mean",
    "Recall_mean",
    "F1_mean"
]].copy()

# Round values
performance_table["Accuracy_mean"] = performance_table["Accuracy_mean"].round(4)
performance_table["Recall_mean"] = performance_table["Recall_mean"].round(4)
performance_table["F1_mean"] = performance_table["F1_mean"].round(4)

# Sort by Accuracy
performance_table = performance_table.sort_values(
    by="Accuracy_mean",
    ascending=False
)

print("\n================ MODEL PERFORMANCE TABLE ================\n")
print(performance_table.to_string(index=False))
print("\n=========================================================\n")
# =====================================================
# Visualization (MEAN METRICS)
# =====================================================
os.makedirs("plots", exist_ok=True)
# Prepare data for combined plot
plot_df = df_results.melt(
    id_vars="Model",
    value_vars=["Accuracy_mean", "Recall_mean", "F1_mean"],
    var_name="Metric",
    value_name="Score"
)

# Clean metric names for display
plot_df["Metric"] = plot_df["Metric"].str.replace("_mean", "")
plt.figure(figsize=(14, 6))
sns.barplot(
    x="Model",
    y="Score",
    hue="Metric",
    data=plot_df,
    palette="viridis"
)

plt.xticks(rotation=45)
plt.ylabel("Score")
plt.ylim(0, 1)
plt.title("Model Comparison: Accuracy vs Recall vs F1 (Stage-1 COPD Detection)")
plt.legend(title="Metric")
plt.tight_layout()
plt.savefig("plots/model_comparison_all_metrics.png")
plt.show()


# =====================================================
# Feature Importance (Random Forest)
# =====================================================
# =====================================================
# Feature Importance (Random Forest - Top 15)
# =====================================================

rf_pipeline = joblib.load("models/saved_models/ExtraTrees.pkl")
rf_model = rf_pipeline.named_steps["model"]

# Build feature names
stat_names = ["Mean", "Std", "Min", "Max", "Median",
              "RMS", "Energy", "Skewness", "Kurtosis"]

feature_names = []
for sensor in range(1, 9):
    for stat in stat_names:
        feature_names.append(f"S{sensor}_{stat}")

fi_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

# 🔥 Show only Top 15
top_n = 15
fi_top = fi_df.head(top_n)

plt.figure(figsize=(8, 6))
sns.barplot(
    x="Importance",
    y="Feature",
    data=fi_top,
    palette="magma"
)

plt.title("Top 15 Important Features (ExtraTrees)")
plt.tight_layout()
plt.savefig("plots/top15_feature_importance_magma.png")
plt.show()