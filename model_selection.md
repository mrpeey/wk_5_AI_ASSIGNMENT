# Model Selection and Justification
## Hospital Patient Readmission Risk Prediction System

## 1. Model Selection Process

### 1.1 Problem Characteristics

- **Task Type**: Binary classification (readmitted vs. not readmitted within 30 days)
- **Data Type**: Mixed (numerical and categorical features)
- **Sample Size**: Medium to large (10,000+ patient records)
- **Class Imbalance**: Yes (typically 15-20% readmission rate)
- **Interpretability Requirement**: High (clinical decision support)
- **Real-time Requirement**: No (batch predictions acceptable)
- **Regulatory Context**: Healthcare (FDA, ONC potential oversight)

### 1.2 Model Requirements

**Must Have:**
1. **High Interpretability**: Clinicians need to understand why a patient is flagged as high-risk
2. **Reliable Probability Estimates**: For risk stratification and resource allocation
3. **Robust to Class Imbalance**: Handle minority class (readmissions) effectively
4. **Fairness**: Consistent performance across demographic groups
5. **Clinical Validity**: Features and predictions align with clinical knowledge

**Nice to Have:**
1. Fast training time for frequent retraining
2. Handle missing data gracefully
3. Capture non-linear relationships
4. Feature importance ranking

---

## 2. Candidate Models Evaluated

### Model 1: Logistic Regression
**Type**: Linear model

**Pros**:
- ✅ Highly interpretable (coefficients = feature importance)
- ✅ Provides well-calibrated probability estimates
- ✅ Fast training and prediction
- ✅ Well-understood in clinical community
- ✅ Robust with proper regularization
- ✅ Works well with many features
- ✅ Easily explainable to stakeholders

**Cons**:
- ❌ Assumes linear relationships
- ❌ May miss complex feature interactions
- ❌ Requires feature engineering for non-linearity
- ❌ Sensitive to feature scaling

**Performance Expectation**: AUC 0.75-0.80

---

### Model 2: Random Forest
**Type**: Ensemble of decision trees

**Pros**:
- ✅ Captures non-linear relationships
- ✅ Handles feature interactions automatically
- ✅ Provides feature importance
- ✅ Robust to outliers
- ✅ Handles missing data (with imputation)
- ✅ No feature scaling required
- ✅ Less prone to overfitting than single decision tree

**Cons**:
- ⚠️ Less interpretable than logistic regression
- ⚠️ Probability estimates may be poorly calibrated
- ❌ Larger model size
- ❌ Slower prediction time
- ⚠️ Can overfit on noisy data

**Performance Expectation**: AUC 0.78-0.83

---

### Model 3: Gradient Boosting (XGBoost/LightGBM)
**Type**: Sequential ensemble of decision trees

**Pros**:
- ✅ Often highest predictive performance
- ✅ Handles non-linear relationships
- ✅ Built-in handling of missing data
- ✅ Feature importance available
- ✅ Handles class imbalance well (with scale_pos_weight)
- ✅ Regularization options (L1, L2)

**Cons**:
- ❌ Black box model (low interpretability)
- ❌ Requires careful hyperparameter tuning
- ❌ Risk of overfitting if not properly tuned
- ❌ Longer training time
- ⚠️ Probability calibration may be needed
- ❌ Harder to explain to clinical stakeholders

**Performance Expectation**: AUC 0.80-0.85

---

### Model 4: Neural Network (Deep Learning)
**Type**: Multi-layer perceptron

**Pros**:
- ✅ Can capture very complex patterns
- ✅ Handles non-linearity
- ✅ Can process mixed data types
- ✅ Scalable to large datasets

**Cons**:
- ❌ Black box (very low interpretability)
- ❌ Requires large amounts of data
- ❌ Computationally expensive
- ❌ Difficult to tune
- ❌ Prone to overfitting on small datasets
- ❌ Hard to explain to clinicians
- ❌ Overkill for tabular data

**Performance Expectation**: AUC 0.78-0.83 (not necessarily better than simpler models for tabular data)

---

### Model 5: Support Vector Machine (SVM)
**Type**: Kernel-based classifier

**Pros**:
- ✅ Effective in high-dimensional spaces
- ✅ Works well with clear margin of separation
- ✅ Memory efficient

**Cons**:
- ❌ Not probabilistic by default (requires Platt scaling)
- ❌ Difficult to interpret (kernel function)
- ❌ Slow training on large datasets
- ❌ Sensitive to feature scaling
- ❌ Requires careful kernel selection

**Performance Expectation**: AUC 0.76-0.80

---

## 3. Model Selection Decision

### **SELECTED MODEL: Gradient Boosting (XGBoost)**

### **Backup Model: Logistic Regression**

---

## 4. Justification for XGBoost

### 4.1 Why XGBoost is the Primary Choice

**1. Superior Predictive Performance**
- Consistently achieves highest AUC across medical prediction tasks
- Handles complex non-linear relationships in healthcare data
- Expected AUC: 0.80-0.85 (vs 0.75-0.80 for logistic regression)
- Even 3-5% improvement in AUC translates to hundreds of patients correctly identified

**2. Robust to Data Challenges**
- **Built-in missing value handling**: Learns optimal direction for missing values
- **Handles class imbalance**: `scale_pos_weight` parameter addresses 15-20% readmission rate
- **Outlier resistant**: Tree-based structure less sensitive to extreme values
- **No feature scaling required**: Simplifies preprocessing

**3. Clinical Relevance**
- **Feature importance**: Identifies most influential predictors (CCI, prior admissions, LOS)
- **SHAP values**: Provides patient-level explanations (e.g., "High risk because: 3 prior admissions + CHF + long LOS")
- **Threshold flexibility**: Clinicians can adjust sensitivity/specificity based on resources
- **Validated in healthcare**: Multiple studies show XGBoost success in readmission prediction

**4. Practical Advantages**
- **Efficient training**: Parallelized computation, trains in minutes even on 100k+ records
- **Production-ready**: Libraries mature (XGBoost, LightGBM) with extensive documentation
- **Regularization**: L1, L2 penalties and early stopping prevent overfitting
- **Handles mixed data**: Seamlessly processes numerical and categorical features

**5. Fairness and Calibration**
- Can apply fairness constraints during training
- Probability calibration via isotonic regression improves reliability
- Monitoring tools can detect bias across demographic groups

### 4.2 Why Not Other Models?

**Logistic Regression**:
- More interpretable but lower performance (75-80% AUC)
- May miss important non-linear patterns (e.g., age × comorbidity interaction)
- However, excellent as backup model for comparison and validation

**Random Forest**:
- Good performance but worse probability calibration than XGBoost
- Larger model size, slower inference
- XGBoost typically outperforms with proper tuning

**Neural Networks**:
- Overkill for tabular data (not better than XGBoost)
- Poor interpretability for clinical context
- Requires much more data and tuning

**SVM**:
- Slower training, worse interpretability
- No built-in probability estimates
- XGBoost consistently outperforms

---

## 5. XGBoost Implementation Details

### 5.1 Hyperparameters

```python
import xgboost as xgb

# Optimal hyperparameters (tuned via cross-validation)
xgb_params = {
    # Model complexity
    'max_depth': 5,                  # Limit tree depth to prevent overfitting
    'min_child_weight': 5,           # Minimum sum of instance weight in a child
    'gamma': 0.1,                    # Minimum loss reduction for split
    
    # Regularization
    'lambda': 1.0,                   # L2 regularization (default)
    'alpha': 0.1,                    # L1 regularization
    
    # Learning rate
    'eta': 0.05,                     # Learning rate (lower = more robust, needs more trees)
    'n_estimators': 300,             # Number of boosting rounds
    
    # Class imbalance (15% readmission rate → weight ratio 85/15 ≈ 5.67)
    'scale_pos_weight': 5.67,        # Balance class weights
    
    # Randomness for robustness
    'subsample': 0.8,                # Sample 80% of data for each tree
    'colsample_bytree': 0.8,         # Sample 80% of features for each tree
    
    # Objective and evaluation
    'objective': 'binary:logistic',  # Binary classification
    'eval_metric': 'auc',            # Area under ROC curve
    
    # System
    'tree_method': 'hist',           # Fast histogram-based method
    'random_state': 42,              # Reproducibility
    'n_jobs': -1                     # Use all CPU cores
}

model = xgb.XGBClassifier(**xgb_params)
```

### 5.2 Training Strategy

**Cross-Validation**:
```python
from sklearn.model_selection import StratifiedKFold

# 5-fold stratified CV (maintains class distribution)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Train with early stopping on validation set
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,        # Stop if no improvement for 50 rounds
    verbose=True
)
```

**Hyperparameter Tuning**:
```python
from sklearn.model_selection import RandomizedSearchCV

# Search space
param_grid = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'eta': [0.01, 0.05, 0.1],
    'scale_pos_weight': [4, 5, 6]
}

# Randomized search (faster than grid search)
random_search = RandomizedSearchCV(
    xgb.XGBClassifier(),
    param_distributions=param_grid,
    n_iter=50,
    scoring='roc_auc',
    cv=5,
    random_state=42
)

random_search.fit(X_train, y_train)
best_params = random_search.best_params_
```

### 5.3 Interpretability via SHAP

```python
import shap

# Calculate SHAP values for model explanations
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global feature importance
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Individual patient explanation
patient_idx = 0
shap.waterfall_plot(shap.Explanation(
    values=shap_values[patient_idx],
    base_values=explainer.expected_value,
    data=X_test.iloc[patient_idx],
    feature_names=X_test.columns.tolist()
))
```

**Example SHAP Explanation**:
```
Patient ID: 12345 - Predicted Readmission Risk: 72%

Top Risk Factors (SHAP values):
+0.15: prior_admissions_1y = 4 (high utilization)
+0.12: charlson_comorbidity_index = 6 (high comorbidity burden)
+0.09: length_of_stay = 14 days (long stay)
+0.07: heart_failure_diagnosis = 1 (high-risk condition)
+0.05: discharge_to_facility = 1 (post-acute care)
-0.02: followup_scheduled = 1 (protective factor)
-0.01: age = 55 (younger than typical)
```

### 5.4 Probability Calibration

```python
from sklearn.calibration import CalibratedClassifierCV

# Calibrate probabilities using isotonic regression
calibrated_model = CalibratedClassifierCV(
    model, 
    method='isotonic',  # Non-parametric calibration
    cv='prefit'         # Use already fitted model
)

calibrated_model.fit(X_val, y_val)

# Use calibrated model for predictions
y_pred_proba_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
```

### 5.5 Fairness Mitigation

```python
# Monitor performance by demographic group
for group in ['White', 'Black', 'Hispanic', 'Asian']:
    mask = (demographic_data['race_ethnicity'] == group)
    group_auc = roc_auc_score(y_test[mask], y_pred_proba[mask])
    group_precision = precision_score(y_test[mask], y_pred[mask])
    group_recall = recall_score(y_test[mask], y_pred[mask])
    
    print(f"{group}: AUC={group_auc:.3f}, Precision={group_precision:.3f}, Recall={group_recall:.3f}")

# If disparities detected (>5% difference), consider:
# 1. Reweighting samples by group
# 2. Using fairness-aware thresholds (different per group)
# 3. Adversarial debiasing
```

---

## 6. Comparison Model: Logistic Regression

### Why Keep Logistic Regression as Backup?

1. **Regulatory/Audit Purposes**: Simpler model easier to validate and audit
2. **Baseline Comparison**: Ensures XGBoost improvement is meaningful
3. **Interpretability**: Easier to explain to non-technical stakeholders
4. **Fallback**: If XGBoost fails or shows unexpected behavior

### Logistic Regression Implementation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Standardize features (required for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic regression with L2 regularization
lr_model = LogisticRegression(
    penalty='l2',
    C=1.0,                          # Regularization strength (inverse)
    class_weight='balanced',        # Handle class imbalance
    max_iter=1000,
    random_state=42,
    solver='lbfgs'
)

lr_model.fit(X_train_scaled, y_train)

# Feature importance via coefficients
feature_importance_lr = pd.DataFrame({
    'feature': feature_names,
    'coefficient': lr_model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)
```

---

## 7. Expected Performance Metrics

### 7.1 Primary Metrics

**XGBoost (Expected)**:
- **AUC-ROC**: 0.82 (95% CI: 0.80-0.84)
- **Precision**: 0.76 (at decision threshold optimized for F1)
- **Recall**: 0.71
- **F1 Score**: 0.73

**Logistic Regression (Expected)**:
- **AUC-ROC**: 0.77 (95% CI: 0.75-0.79)
- **Precision**: 0.70
- **Recall**: 0.68
- **F1 Score**: 0.69

### 7.2 Clinical Impact Metrics

- **Number Needed to Evaluate (NNE)**: ~3 patients flagged to identify 1 true readmission
- **False Alarm Rate**: ~24% (1 in 4 flagged patients won't readmit)
- **Miss Rate**: ~29% (will miss ~3 in 10 actual readmissions)

### 7.3 Fairness Metrics

**Target**: <5% difference across demographic groups
- AUC difference: <0.05
- Precision difference: <0.05
- Recall difference: <0.05

---

## 8. Model Validation Strategy

### 8.1 Temporal Validation
```python
# Train on 2021-2022 data
# Validate on 2023 Q1 data (unseen time period)
# Ensures model generalizes to future patients
```

### 8.2 External Validation
```python
# Train on Hospital A data
# Validate on Hospital B data (different patient population)
# Ensures model generalizes across sites
```

### 8.3 Subgroup Validation
```python
# Validate performance in key subgroups:
# - Heart failure patients
# - COPD patients
# - Elderly (age > 75)
# - High-utilizers (>3 prior admissions)
```

---

## 9. Model Deployment

### 9.1 Production Architecture

```
┌─────────────────┐
│  EHR System     │
│  (Epic/Cerner)  │
└────────┬────────┘
         │ Patient discharge event
         ↓
┌─────────────────────────────────┐
│  Data Extraction Service        │
│  (Extract patient features)     │
└────────┬────────────────────────┘
         │ Raw features (JSON)
         ↓
┌─────────────────────────────────┐
│  Preprocessing Service          │
│  (Apply fitted pipeline)        │
└────────┬────────────────────────┘
         │ Transformed features
         ↓
┌─────────────────────────────────┐
│  XGBoost Model Service          │
│  (Generate prediction + SHAP)   │
└────────┬────────────────────────┘
         │ Risk score + explanation
         ↓
┌─────────────────────────────────┐
│  EHR Integration Service        │
│  (Write back to patient chart)  │
└─────────────────────────────────┘
         │
         ↓
┌─────────────────────────────────┐
│  Clinical Dashboard             │
│  (Display high-risk patients)   │
└─────────────────────────────────┘
```

### 9.2 Model Monitoring

**Automated Monitoring**:
- **Data Drift**: Monitor feature distributions weekly
- **Performance**: Recalculate AUC monthly on new labeled data
- **Fairness**: Check demographic performance disparities monthly
- **Prediction Volume**: Track daily prediction counts

**Alerts**:
- AUC drops below 0.75 → Investigate immediately
- Feature drift exceeds threshold → Review data quality
- Demographic disparity >7% → Fairness review

### 9.3 Model Retraining

**Schedule**:
- **Quarterly retraining**: Incorporate new data
- **Annual full review**: Reassess features and hyperparameters
- **Ad-hoc retraining**: If monitoring detects performance degradation

---

## 10. Conclusion

**XGBoost is selected as the primary model** for hospital readmission prediction because it offers:

1. **Best predictive performance** (AUC 0.80-0.85)
2. **Robustness** to data challenges (missing values, imbalance, outliers)
3. **Clinical interpretability** via SHAP values
4. **Production maturity** and efficiency
5. **Fairness capabilities** with monitoring and mitigation

**Logistic regression serves as a backup model** for:
- Regulatory validation
- Baseline comparison
- Simpler explanations to stakeholders

This dual-model approach balances **performance** (XGBoost) with **transparency** (Logistic Regression), ensuring both clinical effectiveness and stakeholder trust.

The selected model is appropriate for the healthcare context, meeting requirements for accuracy, interpretability, fairness, and production deployment.
