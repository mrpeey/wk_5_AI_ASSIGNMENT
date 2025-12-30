# Data Architecture and Sources
## Hospital Patient Readmission Risk Prediction System

## 1. Proposed Data Sources

### Primary Data Source: Electronic Health Records (EHR)

#### A. Clinical Data
- **Diagnoses**: 
  - Primary and secondary diagnosis codes (ICD-10)
  - Comorbidity indices (Charlson, Elixhauser)
  - Chronic conditions (diabetes, COPD, heart failure, etc.)
  
- **Procedures**:
  - Surgical procedures during admission
  - Diagnostic procedures (imaging, lab tests)
  - CPT codes for procedures performed

- **Medications**:
  - Discharge medication list
  - Changes in medication regimen
  - Polypharmacy indicators (number of medications)
  - High-risk medications (anticoagulants, insulin, etc.)

- **Vital Signs and Lab Results**:
  - Blood pressure, heart rate, temperature
  - Lab values at discharge (creatinine, hemoglobin, glucose, etc.)
  - Abnormal lab flags

- **Clinical Notes**:
  - Discharge summaries
  - Progress notes (via NLP processing)
  - Provider assessments

#### B. Encounter Data
- Length of stay (LOS)
- Admission type (emergency, elective, urgent)
- Admission source (ER, transfer, direct)
- Discharge disposition (home, SNF, rehabilitation)
- ICU stays during admission
- Number of previous admissions (30, 90, 365 days)

### Secondary Data Sources:

#### C. Demographic Data
- Age, gender, race/ethnicity
- Primary language
- Marital status
- Geographic location (zip code, rural/urban)

#### D. Socioeconomic Data
- Insurance type (Medicare, Medicaid, private, uninsured)
- Socioeconomic status indicators (from zip code)
- Employment status
- Housing status

#### E. Post-Discharge Care Data
- Follow-up appointments scheduled
- Home health services arranged
- Durable medical equipment orders
- Patient education materials provided

#### F. External Data Sources
- **Social Determinants of Health (SDOH) databases**:
  - Area Deprivation Index (ADI)
  - Food access data
  - Transportation availability
  
- **Pharmacy Benefits Manager (PBM) Data**:
  - Medication adherence rates
  - Prescription fill patterns

- **Health Information Exchange (HIE)**:
  - Care received at other facilities
  - Emergency department visits at other hospitals

## 2. Data Schema

### Patient Features Table
```
patient_features:
├── patient_id (PRIMARY KEY, encrypted)
├── demographic_features
│   ├── age (integer)
│   ├── gender (categorical)
│   ├── race_ethnicity (categorical)
│   └── primary_language (categorical)
├── clinical_features
│   ├── primary_diagnosis (ICD-10 code)
│   ├── charlson_comorbidity_index (integer)
│   ├── number_of_diagnoses (integer)
│   ├── number_of_procedures (integer)
│   └── discharge_medications_count (integer)
├── encounter_features
│   ├── length_of_stay (days)
│   ├── admission_type (categorical)
│   ├── icu_stay_flag (boolean)
│   ├── prior_admissions_30d (integer)
│   └── prior_admissions_1y (integer)
├── lab_features
│   ├── creatinine_discharge (float)
│   ├── hemoglobin_discharge (float)
│   └── glucose_discharge (float)
├── socioeconomic_features
│   ├── insurance_type (categorical)
│   ├── adi_score (integer, 1-100)
│   └── urban_rural_code (categorical)
└── discharge_planning_features
    ├── follow_up_scheduled_flag (boolean)
    ├── home_health_ordered (boolean)
    └── patient_education_completed (boolean)
```

### Target Variable
```
readmission_target:
├── patient_id (FOREIGN KEY)
├── readmission_30d (boolean: 0=No, 1=Yes)
└── days_to_readmission (integer, NULL if no readmission)
```

## 3. Data Requirements

### Data Quality Requirements:
- **Completeness**: ≥95% for critical fields (age, gender, primary diagnosis, LOS)
- **Accuracy**: Regular validation against source systems
- **Timeliness**: Data available within 4 hours of discharge
- **Consistency**: Standardized coding across all sources

### Data Volume:
- **Training Data**: Minimum 10,000 discharge episodes (with 15-20% readmission rate)
- **Historical Lookback**: 2-3 years of historical data
- **Update Frequency**: Daily batch updates for model retraining

### Data Retention:
- **Active Data**: 3 years readily accessible
- **Archived Data**: 7 years for compliance (HIPAA requirement)
- **De-identified Data**: Indefinite for research purposes

## 4. Data Access and Security

### Access Controls:
- **Role-Based Access Control (RBAC)**: Only authorized personnel
- **Audit Logging**: All data access logged and monitored
- **Encryption**: 
  - At rest: AES-256 encryption
  - In transit: TLS 1.3
  - PHI tokenization for non-clinical users

### Compliance:
- **HIPAA Compliance**: All data handling follows HIPAA Security Rule
- **Data Use Agreements**: Formal agreements with external data sources
- **Business Associate Agreements (BAA)**: With all vendors

## 5. Data Integration Strategy

### ETL Pipeline:
1. **Extract**: 
   - Nightly extracts from EHR (Epic, Cerner, etc.)
   - API integration with HIE
   - Batch files from external sources

2. **Transform**:
   - Data cleaning and validation
   - Feature engineering
   - Standardization (e.g., units, codes)
   - De-duplication

3. **Load**:
   - Load to secure data warehouse
   - Create feature store for model serving
   - Update data quality dashboards

### Data Governance:
- **Data Dictionary**: Comprehensive documentation of all fields
- **Data Lineage**: Track data from source to model prediction
- **Data Quality Monitoring**: Automated checks for anomalies
- **Stewardship**: Designated data owners for each source system

## 6. Feature Engineering Data Sources

Key features derived from raw data:

1. **Comorbidity Burden**: Calculated from diagnosis codes
2. **Medication Complexity**: Derived from medication list
3. **Healthcare Utilization**: From historical encounter data
4. **Social Risk Score**: Composite of SDOH indicators
5. **Discharge Readiness**: Calculated from discharge planning activities
6. **Clinical Instability**: From vital signs and lab trends

This comprehensive data architecture ensures the model has access to all relevant information while maintaining security, privacy, and compliance requirements.
