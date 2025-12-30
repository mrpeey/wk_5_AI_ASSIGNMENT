# Data Preprocessing Pipeline
## Hospital Patient Readmission Risk Prediction System

## Pipeline Overview

This document outlines a comprehensive preprocessing pipeline that transforms raw Electronic Health Record (EHR) data into model-ready features for predicting 30-day readmission risk.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAW DATA SOURCES                            │
│  EHR | Demographics | Labs | Medications | Encounters | SDOH   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   STAGE 1: DATA EXTRACTION                       │
│  • Query EHR database for discharge episodes                    │
│  • Extract patient demographics, clinical data                  │
│  • Pull historical encounter data (lookback 1 year)             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   STAGE 2: DATA VALIDATION                       │
│  • Check data completeness                                       │
│  • Validate data types and ranges                               │
│  • Flag data quality issues                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   STAGE 3: DATA CLEANING                         │
│  • Handle missing values                                         │
│  • Remove duplicates                                            │
│  • Correct erroneous values                                     │
│  • Standardize formats                                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   STAGE 4: FEATURE ENGINEERING                   │
│  • Create derived features                                       │
│  • Calculate comorbidity indices                                │
│  • Aggregate historical data                                    │
│  • Encode categorical variables                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   STAGE 5: FEATURE TRANSFORMATION                │
│  • Scale numerical features                                      │
│  • One-hot encode categorical features                          │
│  • Handle imbalanced classes                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   STAGE 6: FEATURE SELECTION                     │
│  • Remove highly correlated features                            │
│  • Select most important features                               │
│  • Reduce dimensionality if needed                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   MODEL-READY FEATURES                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Data Extraction

### Objective
Extract relevant data from multiple source systems for patients discharged in the target period.

### Steps

1. **Identify Target Population**
   - Inclusion Criteria:
     - All adult patients (age ≥18) discharged from inpatient units
     - Discharge date within extraction period
     - Alive at discharge
   - Exclusion Criteria:
     - Patients discharged to hospice (different prediction problem)
     - Patients who died during admission
     - Obstetric patients (unique readmission patterns)
     - Psychiatric primary admissions (separate model recommended)

2. **Extract Patient Demographics**
   ```
   - patient_id (anonymized)
   - age_at_discharge
   - gender
   - race_ethnicity
   - primary_language
   - marital_status
   - insurance_type
   ```

3. **Extract Clinical Data**
   ```
   - Primary diagnosis (ICD-10)
   - Secondary diagnoses (all ICD-10 codes)
   - Procedures performed (CPT codes)
   - Discharge medications (RxNorm codes, count)
   - Discharge vital signs (last recorded)
   - Discharge lab values (last 24 hours)
   ```

4. **Extract Encounter Data**
   ```
   - admission_date
   - discharge_date
   - admission_source (ER, transfer, elective)
   - admission_type (emergency, urgent, elective)
   - discharge_disposition
   - ICU_days
   - Historical admissions (past 30, 90, 365 days)
   - Historical ED visits (past 365 days)
   ```

5. **Extract Social Determinants**
   ```
   - Zip code (for area-level SDOH)
   - Employment status
   - Housing status
   - Transportation access
   ```

6. **Define Target Variable**
   ```
   - readmission_30d: 1 if patient readmitted to any hospital within 30 days, 0 otherwise
   - Exclude planned readmissions (oncology, obstetrics)
   ```

---

## Stage 2: Data Validation

### Objective
Ensure data quality and identify issues before processing.

### Validation Checks

1. **Completeness Check**
   ```python
   Required fields (must be ≥95% complete):
   - patient_id: 100%
   - age_at_discharge: 100%
   - gender: 100%
   - primary_diagnosis: 100%
   - length_of_stay: 100%
   - admission_type: 100%
   
   Important fields (must be ≥80% complete):
   - race_ethnicity: 80%
   - insurance_type: 80%
   - discharge_medications_count: 80%
   ```

2. **Range Validation**
   ```python
   - age_at_discharge: [18, 120]
   - length_of_stay: [0, 365]
   - num_medications: [0, 50]
   - systolic_bp: [60, 250]
   - diastolic_bp: [30, 150]
   - heart_rate: [20, 200]
   - temperature_f: [95, 108]
   ```

3. **Data Type Validation**
   - Ensure dates are in correct format (YYYY-MM-DD)
   - Numeric fields contain valid numbers
   - Categorical fields contain expected values

4. **Logical Consistency**
   ```python
   - discharge_date >= admission_date
   - age_at_discharge >= 18
   - length_of_stay = discharge_date - admission_date
   - If ICU_days > 0, then ICU_flag = True
   ```

5. **Duplicate Detection**
   - Check for duplicate patient_id within same extraction period
   - Resolve by keeping most recent record

### Output
- Data quality report
- Flagged records for review
- Decision: Proceed or halt pipeline

---

## Stage 3: Data Cleaning

### Objective
Handle missing data, outliers, and inconsistencies.

### 3.1 Missing Value Handling

#### Strategy by Feature Type:

**A. Critical Clinical Features (cannot be missing)**
- `age_at_discharge`: **Exclude record** if missing (should be 100% complete)
- `primary_diagnosis`: **Exclude record** if missing
- `length_of_stay`: **Exclude record** if missing

**B. Important Clinical Features**
- `secondary_diagnoses`: Fill missing with 0 (patient may have no comorbidities)
- `num_medications`: Fill missing with median value
- `lab_values` (creatinine, hemoglobin, etc.):
  - Fill with population median by age group and gender
  - Create binary flag: `lab_value_missing` = 1 if missing

**C. Demographic Features**
- `race_ethnicity`: Fill missing with "Unknown" category
- `marital_status`: Fill missing with "Unknown" category
- `primary_language`: Fill missing with "English" (most common)

**D. Historical Utilization Features**
- `prior_admissions_30d`, `prior_admissions_1y`: Fill missing with 0 (assume no prior admissions)
- `prior_ed_visits_1y`: Fill missing with 0

**E. Social Determinants**
- `adi_score` (area deprivation index): Fill with national median
- `employment_status`: Fill missing with "Unknown"

**F. Create Missing Indicators**
```python
# Create binary flags for important missing features
missing_features = [
    'creatinine_missing',
    'hemoglobin_missing',
    'glucose_missing',
    'race_ethnicity_unknown'
]
```

### 3.2 Outlier Detection and Treatment

**A. Statistical Outliers**
```python
# Use IQR method for continuous variables
Q1 = quantile(feature, 0.25)
Q3 = quantile(feature, 0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 3 * IQR  # Use 3 IQR for medical data (more conservative)
upper_bound = Q3 + 3 * IQR

# Treatment:
# - Cap values at bounds (Winsorization) rather than removing
# - Preserve extreme but plausible values
```

**B. Clinical Outliers**
```python
# Apply clinical knowledge:
# - Length of stay > 30 days: Keep but flag
# - Num medications > 25: Keep but cap at 25 (extreme polypharmacy)
# - Age > 110: Review individually
```

### 3.3 Error Correction

**A. Standardize Formats**
```python
# Date formats: Convert all to YYYY-MM-DD
# Gender: Standardize to M, F, Other
# Race/Ethnicity: Map to standard categories (OMB categories)
```

**B. Code Mapping**
```python
# ICD-10 codes: Ensure valid format (A00.0)
# Map ICD-9 to ICD-10 if historical data included
# Standardize medication names (map to RxNorm)
```

**C. Remove Duplicates**
```python
# Remove exact duplicates: Same patient_id, admission_date, discharge_date
# Keep most complete record if duplicates found
```

### 3.4 Data Cleaning Output
- Clean dataset with documented transformations
- Cleaning report: Number of records/features affected
- Preserved raw data for audit trail

---

## Stage 4: Feature Engineering

### Objective
Create clinically meaningful features from raw data to improve model performance.

### 4.1 Comorbidity Features

**A. Charlson Comorbidity Index (CCI)**
```python
# Calculate CCI score from ICD-10 diagnosis codes
# Weights:
# - Myocardial infarction: 1
# - Congestive heart failure: 1
# - Peripheral vascular disease: 1
# - Dementia: 1
# - COPD: 1
# - Diabetes without complications: 1
# - Diabetes with complications: 2
# - Renal disease: 2
# - Cancer: 2
# - Metastatic cancer: 6
# - Severe liver disease: 3

cci_score = sum_of_condition_weights
cci_category = cut(cci_score, bins=[0, 2, 4, inf], labels=['Low', 'Medium', 'High'])
```

**B. Elixhauser Comorbidity Indicators**
```python
# Binary indicators for 30 comorbidity categories
# Example categories:
elixhauser_features = [
    'chf_flag',              # Congestive heart failure
    'arrhythmia_flag',       # Cardiac arrhythmias
    'valve_disease_flag',    # Valvular disease
    'pulm_circ_flag',        # Pulmonary circulation disorders
    'pvd_flag',              # Peripheral vascular disorders
    'hypertension_flag',     # Hypertension
    'diabetes_uncomplicated_flag',
    'diabetes_complicated_flag',
    'hypothyroidism_flag',
    'renal_failure_flag',
    'liver_disease_flag',
    'copd_flag',
    'obesity_flag',
    'depression_flag',
    'psychoses_flag'
]
```

**C. Number of Diagnoses**
```python
num_diagnoses = count(unique_icd10_codes)
diagnosis_complexity = 'High' if num_diagnoses > 10 else 'Low'
```

### 4.2 Medication Features

**A. Medication Count and Complexity**
```python
num_discharge_medications = count(discharge_medications)
polypharmacy_flag = 1 if num_discharge_medications >= 5 else 0
extreme_polypharmacy_flag = 1 if num_discharge_medications >= 10 else 0
```

**B. High-Risk Medication Indicators**
```python
# Binary flags for high-risk medications
high_risk_meds = {
    'anticoagulant_flag': ['warfarin', 'apixaban', 'rivaroxaban'],
    'insulin_flag': ['insulin'],
    'opioid_flag': ['oxycodone', 'morphine', 'hydrocodone'],
    'diuretic_flag': ['furosemide', 'hydrochlorothiazide']
}
```

**C. Medication Changes**
```python
num_new_medications = count(medications_not_in_admission_list)
medication_change_flag = 1 if num_new_medications > 0 else 0
```

### 4.3 Healthcare Utilization Features

**A. Historical Admission Features**
```python
# Count admissions in various lookback periods
prior_admissions_30d = count(admissions in past 30 days)
prior_admissions_90d = count(admissions in past 90 days)
prior_admissions_1y = count(admissions in past 365 days)

# Frequency
admission_frequency = prior_admissions_1y / 365  # admissions per day

# Binary flag for frequent flyer
frequent_flyer_flag = 1 if prior_admissions_1y >= 3 else 0
```

**B. Emergency Department Utilization**
```python
prior_ed_visits_1y = count(ED visits in past 365 days)
ed_frequent_user_flag = 1 if prior_ed_visits_1y >= 4 else 0
```

**C. Days Since Last Admission**
```python
days_since_last_admission = current_admission_date - last_discharge_date
# If no prior admission, set to 999 (arbitrary large number)
recent_admission_flag = 1 if days_since_last_admission <= 30 else 0
```

### 4.4 Length of Stay Features

**A. Length of Stay Metrics**
```python
length_of_stay = discharge_date - admission_date  # in days

# Categorize
los_category = cut(length_of_stay, 
                   bins=[0, 3, 7, 14, inf], 
                   labels=['Short', 'Medium', 'Long', 'VeryLong'])

# Short stay flag (may indicate premature discharge)
short_stay_flag = 1 if length_of_stay < 3 else 0
```

**B. ICU Utilization**
```python
icu_flag = 1 if icu_days > 0 else 0
icu_percentage = icu_days / length_of_stay if length_of_stay > 0 else 0
```

### 4.5 Lab Value Features

**A. Abnormal Lab Flags**
```python
# Define normal ranges
normal_ranges = {
    'creatinine': (0.6, 1.2),  # mg/dL
    'hemoglobin': (12, 17),    # g/dL
    'glucose': (70, 140),      # mg/dL
    'sodium': (135, 145),      # mEq/L
    'potassium': (3.5, 5.0)    # mEq/L
}

# Create binary flags
creatinine_abnormal = 1 if creatinine outside normal_range else 0
hemoglobin_abnormal = 1 if hemoglobin outside normal_range else 0
# ... repeat for other labs
```

**B. Derived Lab Features**
```python
# eGFR (estimated glomerular filtration rate) - kidney function
# Using CKD-EPI equation
egfr = calculate_egfr(creatinine, age, gender, race)
egfr_category = cut(egfr, 
                    bins=[0, 30, 60, 90, inf],
                    labels=['Severe_CKD', 'Moderate_CKD', 'Mild_CKD', 'Normal'])

# Anemia severity
anemia_severity = cut(hemoglobin,
                      bins=[0, 8, 10, 12, inf],
                      labels=['Severe', 'Moderate', 'Mild', 'None'])
```

### 4.6 Diagnosis-Specific Features

**A. Primary Diagnosis Category**
```python
# Map ICD-10 codes to major categories
primary_dx_category = map_icd10_to_category(primary_diagnosis)
# Categories: Cardiovascular, Respiratory, Gastrointestinal, 
#             Infectious, Injury, Other

# High-risk diagnosis flags
high_risk_diagnoses = {
    'heart_failure_primary': ['I50.x'],
    'copd_primary': ['J44.x'],
    'pneumonia_primary': ['J18.x'],
    'sepsis_primary': ['A41.x']
}
```

### 4.7 Admission Characteristics

**A. Admission Source and Type**
```python
# Emergency admission flag
emergency_admission = 1 if admission_type == 'Emergency' else 0

# Admitted via ED flag
admitted_via_ed = 1 if admission_source == 'Emergency Department' else 0

# Transfer from another facility
transfer_flag = 1 if admission_source == 'Transfer' else 0
```

**B. Discharge Disposition**
```python
# Home discharge (reference category)
discharge_to_home = 1 if disposition == 'Home' else 0

# Post-acute care discharge (SNF, Rehab, LTAC)
discharge_to_facility = 1 if disposition in ['SNF', 'Rehab', 'LTAC'] else 0

# Home health services
home_health_services = 1 if disposition == 'Home Health' else 0
```

### 4.8 Social Determinant Features

**A. Area Deprivation Index (ADI)**
```python
# Link zip code to ADI (1-100, higher = more deprived)
adi_score = get_adi_from_zipcode(zipcode)

adi_category = cut(adi_score,
                   bins=[0, 33, 66, 100],
                   labels=['Low_Deprivation', 'Medium_Deprivation', 'High_Deprivation'])

high_deprivation_flag = 1 if adi_score > 66 else 0
```

**B. Insurance Type Categories**
```python
# Medicare (elderly/disabled)
medicare_flag = 1 if insurance == 'Medicare' else 0

# Medicaid (low-income)
medicaid_flag = 1 if insurance == 'Medicaid' else 0

# Uninsured
uninsured_flag = 1 if insurance == 'Self-Pay' else 0
```

**C. Urban/Rural Classification**
```python
# Link zip code to rural-urban continuum code (RUCC)
rural_flag = 1 if RUCC in [7, 8, 9] else 0  # Non-metropolitan
```

### 4.9 Discharge Planning Features

**A. Follow-up Care Indicators**
```python
# Follow-up appointment scheduled before discharge
followup_scheduled = 1 if followup_appointment_date is not None else 0

# Days to follow-up
days_to_followup = followup_appointment_date - discharge_date
early_followup = 1 if days_to_followup <= 7 else 0

# Home health services ordered
home_health_ordered = 1 if home_health_order exists else 0

# Patient education completed
patient_education_flag = 1 if education_checklist_complete else 0
```

### 4.10 Temporal Features

**A. Seasonality**
```python
discharge_month = month(discharge_date)
discharge_season = get_season(discharge_month)  # Winter, Spring, Summer, Fall

# Winter months (higher infection risk)
winter_discharge = 1 if discharge_month in [12, 1, 2] else 0
```

**B. Day of Week**
```python
discharge_day_of_week = day_of_week(discharge_date)

# Weekend discharge (may have less support services)
weekend_discharge = 1 if discharge_day_of_week in ['Saturday', 'Sunday'] else 0
```

### 4.11 Composite Risk Scores

**A. LACE Score**
```python
# LACE Score: Length of stay, Acuity, Comorbidity, ED visits
# Predicts readmission risk

# L: Length of stay points
if los >= 14:
    L = 7
elif los >= 7:
    L = 5
elif los >= 4:
    L = 4
# ... (full scoring logic)

# A: Acuity (admission via ED)
A = 3 if admitted_via_ed else 0

# C: Comorbidity (Charlson)
if cci >= 4:
    C = 5
elif cci == 3:
    C = 3
# ... (full scoring logic)

# E: ED visits in past 6 months
if prior_ed_visits_6mo >= 4:
    E = 4
elif prior_ed_visits_6mo == 3:
    E = 3
# ... (full scoring logic)

lace_score = L + A + C + E  # Range: 0-19
high_lace_score = 1 if lace_score >= 10 else 0
```

**B. HOSPITAL Score**
```python
# HOSPITAL Score: Another validated readmission risk tool
# Hemoglobin, Discharge from Oncology, Sodium, Procedure during admission,
# Index admission Type, Admissions in last year, Length of stay

hospital_score = calculate_hospital_score(
    hemoglobin,
    oncology_service,
    sodium,
    procedure_flag,
    admission_type,
    prior_admissions_1y,
    length_of_stay
)
```

### 4.12 Interaction Features

**A. Clinically Meaningful Interactions**
```python
# Age × Comorbidity burden
age_x_cci = age_at_discharge * cci_score

# Polypharmacy × Renal function
polypharmacy_x_renal = num_medications * (1 if egfr < 60 else 0)

# Frequent utilization × Social deprivation
frequent_flyer_x_adi = frequent_flyer_flag * high_deprivation_flag
```

### Feature Engineering Summary

Total engineered features: ~100-150

**Feature Categories**:
- Comorbidity: 20-30 features
- Medications: 10 features
- Utilization: 10 features
- Labs: 15 features
- Demographics: 10 features
- Social determinants: 8 features
- Discharge planning: 5 features
- Temporal: 5 features
- Composite scores: 5 features
- Interactions: 10 features

---

## Stage 5: Feature Transformation

### Objective
Transform features into formats suitable for machine learning algorithms.

### 5.1 Encoding Categorical Variables

**A. One-Hot Encoding**
```python
# Apply to nominal categorical variables (no order)
one_hot_features = [
    'primary_dx_category',      # Cardiovascular, Respiratory, etc.
    'admission_source',         # ED, Transfer, Direct
    'discharge_disposition',    # Home, SNF, Rehab, etc.
    'race_ethnicity',          # White, Black, Hispanic, Asian, Other
    'insurance_type',          # Medicare, Medicaid, Private, Uninsured
    'discharge_season'         # Winter, Spring, Summer, Fall
]

# Example: primary_dx_category with 6 categories → 6 binary features
# primary_dx_cardiovascular, primary_dx_respiratory, etc.
```

**B. Ordinal Encoding**
```python
# Apply to ordinal categorical variables (natural order)
ordinal_mappings = {
    'los_category': {'Short': 0, 'Medium': 1, 'Long': 2, 'VeryLong': 3},
    'cci_category': {'Low': 0, 'Medium': 1, 'High': 2},
    'adi_category': {'Low_Deprivation': 0, 'Medium_Deprivation': 1, 'High_Deprivation': 2},
    'egfr_category': {'Normal': 0, 'Mild_CKD': 1, 'Moderate_CKD': 2, 'Severe_CKD': 3}
}
```

**C. Target Encoding (for high-cardinality features)**
```python
# For features like zip_code with many unique values
# Replace with mean readmission rate for that category
zip_code_encoded = zip_code.map(zip_code_readmission_rates)
```

### 5.2 Scaling Numerical Features

**A. Standardization (Z-score scaling)**
```python
# Apply to features with normal or near-normal distribution
# Transform to mean=0, std=1

standardize_features = [
    'age_at_discharge',
    'cci_score',
    'num_medications',
    'creatinine',
    'hemoglobin',
    'glucose',
    'egfr'
]

# Formula: z = (x - mean) / std
```

**B. Min-Max Scaling**
```python
# Apply to features with known bounds
# Transform to [0, 1] range

minmax_features = [
    'length_of_stay',
    'icu_days',
    'prior_admissions_1y',
    'adi_score'
]

# Formula: x_scaled = (x - x_min) / (x_max - x_min)
```

**C. Robust Scaling**
```python
# Use for features with outliers
# Uses median and IQR instead of mean and std

robust_features = [
    'days_since_last_admission',
    'prior_ed_visits_1y'
]

# Formula: x_scaled = (x - median) / IQR
```

### 5.3 Handle Class Imbalance

**Typical readmission rate: 15-20% (imbalanced)**

**A. SMOTE (Synthetic Minority Over-sampling Technique)**
```python
# Generate synthetic examples of minority class (readmissions)
# Apply only to training set, not validation/test

from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

**B. Class Weights**
```python
# Adjust model loss function to penalize minority class errors more
# For 20% readmission rate:
class_weight = {0: 1.0, 1: 4.0}  # 4x weight for readmission class
```

**C. Undersampling (Alternative)**
```python
# Randomly undersample majority class
# Use only if large dataset (>100k samples)
```

### 5.4 Feature Transformation Summary

**Output**:
- Encoded categorical features (binary or numeric)
- Scaled numerical features (standardized distribution)
- Balanced training set (if using SMOTE)
- Transformation parameters saved (for applying to new data)

---

## Stage 6: Feature Selection

### Objective
Reduce dimensionality and remove redundant or irrelevant features.

### 6.1 Correlation Analysis

**A. Remove Highly Correlated Features**
```python
# Calculate pairwise correlations
correlation_matrix = X.corr()

# Identify pairs with correlation > 0.85
high_corr_pairs = find_pairs(correlation_matrix > 0.85)

# Remove one feature from each pair (keep more clinically meaningful one)
# Example: If 'num_diagnoses' and 'cci_score' are highly correlated, keep 'cci_score'
```

### 6.2 Variance Threshold

```python
# Remove features with very low variance (near-constant)
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X_reduced = selector.fit_transform(X)
```

### 6.3 Feature Importance Methods

**A. Tree-Based Feature Importance**
```python
# Train Random Forest to get feature importances
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Keep top N features (e.g., top 50)
top_features = feature_importance.head(50)['feature'].tolist()
```

**B. Recursive Feature Elimination (RFE)**
```python
# Iteratively remove least important features
from sklearn.feature_selection import RFE
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=50)
rfe.fit(X_train, y_train)
selected_features = X_train.columns[rfe.support_]
```

**C. L1 Regularization (Lasso)**
```python
# Train model with L1 penalty
# Automatically drives some feature coefficients to zero
from sklearn.linear_model import LogisticRegression
lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)
lasso.fit(X_train, y_train)

# Select features with non-zero coefficients
selected_features = X_train.columns[lasso.coef_[0] != 0]
```

### 6.4 Clinical Validation

**Essential Features (Must Include)**
- Age
- Primary diagnosis category
- Comorbidity index (CCI or Elixhauser)
- Number of prior admissions
- Length of stay
- Key lab values (creatinine, hemoglobin)

### 6.5 Final Feature Set

**Target: 40-60 features**

Typical final feature set includes:
- Demographics: 5-8 features
- Comorbidities: 10-15 features
- Medications: 5-8 features
- Utilization history: 5-8 features
- Labs: 8-10 features
- Social determinants: 3-5 features
- Discharge planning: 3-5 features
- Composite scores: 2-3 features

---

## Pipeline Implementation

### Code Structure

```python
class ReadmissionPreprocessingPipeline:
    def __init__(self):
        self.imputers = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_selector = None
        
    def fit(self, X_train, y_train):
        """Fit preprocessing steps on training data"""
        # Stage 2: Validation
        self._validate_data(X_train)
        
        # Stage 3: Cleaning
        X_clean = self._clean_data(X_train)
        
        # Stage 4: Feature Engineering
        X_engineered = self._engineer_features(X_clean)
        
        # Stage 5: Transformation
        X_transformed = self._transform_features(X_engineered)
        
        # Stage 6: Feature Selection
        X_selected = self._select_features(X_transformed, y_train)
        
        return X_selected
    
    def transform(self, X_new):
        """Apply fitted preprocessing to new data"""
        X_clean = self._clean_data(X_new)
        X_engineered = self._engineer_features(X_clean)
        X_transformed = self._transform_features(X_engineered)
        X_selected = self._select_features(X_transformed)
        return X_selected
```

### Pipeline Persistence

```python
import joblib

# Save fitted pipeline
joblib.dump(preprocessing_pipeline, 'preprocessing_pipeline.pkl')

# Load for inference
preprocessing_pipeline = joblib.load('preprocessing_pipeline.pkl')
```

---

## Quality Assurance

### Monitoring and Validation

1. **Data Drift Detection**
   - Monitor feature distributions over time
   - Alert if new data significantly differs from training data

2. **Feature Value Ranges**
   - Validate that new data falls within expected ranges
   - Flag outliers for review

3. **Pipeline Performance**
   - Track preprocessing execution time
   - Monitor for failures or errors

4. **Documentation**
   - Maintain changelog of pipeline modifications
   - Document all transformations and rationale

---

## Conclusion

This preprocessing pipeline transforms raw EHR data into high-quality, model-ready features through systematic cleaning, engineering, and transformation. The pipeline is:

- **Reproducible**: Same preprocessing applied to training and inference data
- **Maintainable**: Modular design allows easy updates
- **Clinically Informed**: Features engineered based on clinical knowledge
- **Production-Ready**: Handles edge cases and monitors data quality

The resulting feature set balances clinical relevance, predictive power, and model interpretability, setting the foundation for an accurate and trustworthy readmission prediction model.
