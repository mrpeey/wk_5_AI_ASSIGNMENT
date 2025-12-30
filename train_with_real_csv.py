import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Update this path to your real CSV file
CSV_PATH = r'C:\Users\poulo\Downloads\archive (2)\hospital data analysis.csv'

# Load real data from CSV
def load_data(csv_path):
    data = pd.read_csv(csv_path)
    # Example: update these columns to match your CSV
    features = ['age', 'num_comorbidities', 'lab_trend', 'discharge_home']
    target = 'readmitted_30d'
    if not all(col in data.columns for col in features + [target]):
        raise ValueError(f"CSV must contain columns: {features + [target]}")
    X = data[features]
    y = data[target]
    return X, y

if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print(f"CSV file not found: {CSV_PATH}")
    else:
        X, y = load_data(CSV_PATH)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, 'readmission_model.joblib')
        print("Model trained and saved with real data.")
