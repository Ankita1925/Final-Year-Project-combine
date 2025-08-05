
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

def main():
    df = pd.read_csv("lung_disease_data.csv")

    print(df.head())
    print(df.describe())

    gender_mapping ={"Male" : int(1) , "Female" : int(2)}
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map(gender_mapping)

    else:
        print("Gender Not Found")

    Smoking_Status_mapping ={"Yes" : int(1) , "No" : int(0)}
    df['Smoking Status'] = df['Smoking Status'].map(Smoking_Status_mapping)

    Treatment_mapping ={ "Medication" : int(0),"Therapy" : int(1) , "Surgery" : int(2)}
    df['Treatment Type'] = df['Treatment Type'].map(Treatment_mapping)
    

    Disease_Type_mapping ={"Asthma" : int(0) , "COPD" : int(1) , "Lung Cancer" : int(2) , "Pneumonia" : int(3),"Bronchitis": int(4)}
    df['Disease Type'] = df['Disease Type'].map(Disease_Type_mapping)

    recovery_mapping ={"Yes" : int(1) , "No" : int(0)}
    df['Recovered'] = df['Recovered'].map(recovery_mapping)
    
    print(df.head())

    # Check for missing values
    print(df.isnull().sum())

    df.dropna(how = 'any',axis= 0,inplace=True)
    print(df.isnull().sum())

    df.info()

    corr_matrix = df.corr()
    print(corr_matrix)


    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap= "coolwarm")
    plt.title("Correlation Matrix")
    plt.show()


    

    # Select correlation with target column (e.g., 'Recovered')
    target_corr = corr_matrix["Recovered"].drop("Recovered")  # Drop self-correlation

    # Bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(y=target_corr.index, x=target_corr.values)

    plt.title("Feature Correlation with Recovery")
    plt.ylabel("Correlation Coefficient")
    plt.xlabel("Features")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    

if(__name__ == "__main__"):
    main()