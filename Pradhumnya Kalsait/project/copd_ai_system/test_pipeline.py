# test_pipeline.py

from services.pipeline_service import full_copd_pipeline

clinical_input = {
    "Age": 60,
    "Gender": 1,
    "BMI, kg/m2": 27,
    "Height/m": 1.7,
    "History of Heart Failure": 0,
    "working place": 2,
    "mMRC": 3,
    "status of smoking": 1,
    "Pack History": 20,
    "Vaccination": 1,
    "Depression": 0,
    "Dependent": 0,
    "Temperature": 37,
    "Respiratory Rate": 20,
    "Heart Rate": 85,
    "Blood pressure": 2,
    "Oxygen Saturation": 0.94,
    "Sputum": 1
}

result = full_copd_pipeline(
    "C:/Users/Admin/Desktop/Data_Samples/COPD_records/COPD1.csv",
    clinical_input
)

print(result)