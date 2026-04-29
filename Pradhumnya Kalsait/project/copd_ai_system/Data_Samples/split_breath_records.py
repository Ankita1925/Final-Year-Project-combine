import os
import pandas as pd


def split_breath_records(csv_file_path):
    """
    Splits E-Nose dataset into individual patient breath files.
    
    Assumption:
    - Rows = time samples (~4000)
    - Columns = 8 sensors × number_of_breaths
    """

    # Load dataset
    df = pd.read_csv(csv_file_path)

    dataset_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    output_folder = f"{dataset_name}_records"

    os.makedirs(output_folder, exist_ok=True)

    total_columns = df.shape[1]
    n_sensors = 8

    if total_columns % n_sensors != 0:
        raise ValueError("Column count is not divisible by 8 sensors.")

    n_breaths = total_columns // n_sensors

    print(f"\nDataset: {dataset_name}")
    print(f"Total Columns: {total_columns}")
    print(f"Detected Breaths: {n_breaths}")
    print(f"Creating folder: {output_folder}\n")

    for i in range(n_breaths):
        start_col = i * n_sensors
        end_col = (i + 1) * n_sensors

        breath_df = df.iloc[:, start_col:end_col]

        file_name = f"{dataset_name}{i+1}.csv"
        file_path = os.path.join(output_folder, file_name)

        breath_df.to_csv(file_path, index=False)

        print(f"Saved: {file_name}")

    print("\nAll patient breath samples extracted successfully.\n")


# =========================
# Run Script
# =========================
if __name__ == "__main__":
    file_path = input("Enter CSV file name (e.g., SMOKERS.csv): ")
    split_breath_records(file_path)