import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load real data from CSV (update 'hospital_data.csv' to your actual file)
def load_data(csv_path):
    data = pd.read_csv(csv_path)
    # Example preprocessing (customize as needed)
    features = ['age', 'num_comorbidities', 'lab_trend', 'discharge_home']
    X = data[features]
    y = data['readmitted_30d']
    return X, y

if __name__ == "__main__":
    X, y = load_data('hospital_data.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'readmission_model.joblib')
    print("Model trained and saved with real data.")
