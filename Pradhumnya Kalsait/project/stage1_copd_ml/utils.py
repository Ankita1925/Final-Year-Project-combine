import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# =====================================================
# Label Mapping
# =====================================================
LABEL_MAP = {
    "COPD": 0,
    "SMOKERS": 1,
    "CONTROL": 2,
    "AIR": 3
}

# =====================================================
# Feature Extraction (PER SENSOR)
# =====================================================
def extract_features(signal: np.ndarray) -> np.ndarray:
    return np.array([
        np.mean(signal),
        np.std(signal),
        np.min(signal),
        np.max(signal),
        np.median(signal),
        np.sqrt(np.mean(signal ** 2)),  # RMS
        np.sum(signal ** 2),            # Energy
        skew(signal),
        kurtosis(signal)
    ])

# =====================================================
# Correct Loader (8 Sensors per Breath)
# =====================================================
def load_txt_file(path: str, label: str):

    # Load CSV (header auto handled)
    df = pd.read_csv(path)
    raw_data = df.values  # shape: (4000, total_columns)

    n_sensors = 8
    total_columns = raw_data.shape[1]

    if total_columns % n_sensors != 0:
        raise ValueError("Column count is not divisible by 8 sensors!")

    n_breaths = total_columns // n_sensors

    features_list = []

    for i in range(n_breaths):

        # Extract 8 columns for this breath
        breath_block = raw_data[:, i*n_sensors:(i+1)*n_sensors]

        breath_features = []

        # Extract features per sensor
        for s in range(n_sensors):
            sensor_signal = breath_block[:, s]
            breath_features.extend(extract_features(sensor_signal))

        features_list.append(breath_features)

    X = np.array(features_list)
    y = np.full(X.shape[0], LABEL_MAP[label])

    return X, y