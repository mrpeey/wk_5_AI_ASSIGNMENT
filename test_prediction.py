import joblib
import numpy as np

# Example patient data (update values as needed)
patient = {
    'Age': 60,
    'Gender': 1,  # 0=Female, 1=Male (check encoding)
    'Condition': 2,  # integer from label encoding
    'Procedure': 3,  # integer from label encoding
    'Cost': 10000,
    'Length_of_Stay': 5
}

# Load model
model = joblib.load('readmission_model.joblib')

# Prepare input for prediction
features = np.array([[patient['Age'], patient['Gender'], patient['Condition'], patient['Procedure'], patient['Cost'], patient['Length_of_Stay']]])
prediction = model.predict(features)[0]
probability = model.predict_proba(features)[0, 1]

print(f"Predicted Readmission: {prediction} (1=Yes, 0=No)")
print(f"Probability of Readmission: {probability:.2f}")
