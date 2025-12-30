import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# 1. Load Data (placeholder for real hospital data)
data = pd.DataFrame({
    'age': np.random.randint(20, 90, 100),
    'num_comorbidities': np.random.randint(0, 5, 100),
    'lab_trend': np.random.randn(100),
    'discharge_home': np.random.randint(0, 2, 100),
    'readmitted_30d': np.random.randint(0, 2, 100)
})

# 2. Preprocessing
features = ['age', 'num_comorbidities', 'lab_trend', 'discharge_home']
X = data[features]
y = data['readmitted_30d']

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Training
model = RandomForestClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluation
preds = model.predict(X_test)
cm = confusion_matrix(y_test, preds)
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)

print('Confusion Matrix:\n', cm)
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')

# 6. Cross-validation for overfitting check
cv_scores = cross_val_score(model, X, y, cv=5)
print(f'CV Accuracy: {cv_scores.mean():.2f} +/- {cv_scores.std():.2f}')

# 7. (Optional) Save model for deployment
import joblib
joblib.dump(model, 'readmission_model.joblib')
