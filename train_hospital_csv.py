import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

CSV_PATH = r'C:\Users\poulo\Downloads\archive (2)\hospital data analysis.csv'

# Load and preprocess data
def load_data(csv_path):
    data = pd.read_csv(csv_path)
    # Encode categorical columns
    le_gender = LabelEncoder()
    le_condition = LabelEncoder()
    le_procedure = LabelEncoder()
    data['Gender'] = le_gender.fit_transform(data['Gender'])
    data['Condition'] = le_condition.fit_transform(data['Condition'])
    data['Procedure'] = le_procedure.fit_transform(data['Procedure'])
    # Encode target
    data['Readmission'] = data['Readmission'].map({'Yes': 1, 'No': 0})
    # Features and target
    features = ['Age', 'Gender', 'Condition', 'Procedure', 'Cost', 'Length_of_Stay']
    X = data[features]
    y = data['Readmission']
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
        print("Model trained and saved with real hospital data.")
