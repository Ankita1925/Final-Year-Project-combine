import joblib
import pandas as pd

ML = r"D:\Be project\GUI-FYP\backend\ml_models"
model1 = joblib.load(ML + r"\cirhosis\VotingClfCirh.pkl")

# Check class distribution the model was trained on
print("Classes:", model1.classes_)

# Check each sub-model's prediction on healthy values
test = pd.DataFrame([{
    "Age": 35, "Sex": 0,
    "ALB": 450.0, "ALP": 60.0, "ALT": 25.0, "AST": 20.0,
    "BIL": 10.0, "CHE": 8.0, "CHOL": 5.0,
    "CREA": 80.0, "GGT": 20.0, "PROT": 720.0,
}])

for est in model1.estimators_:
    print(f"{est.__class__.__name__}: {est.predict(test)[0]}")