# wk_5_AI_ASSIGNMENT

## Hospital Readmission AI System

### 1. Train Model with Real Data
Run the training script using your CSV:

```
python train_hospital_csv.py
```

### 2. Print Encoding Mappings
To convert categorical values for prediction, print encodings:

```
python print_encodings.py
```

### 3. Test a Prediction
Edit `test_prediction.py` with encoded patient values and run:

```
python test_prediction.py
```

### 4. API Usage
Start the FastAPI server:

```
uvicorn api:app --reload
```

Send POST requests to `/predict` with patient data (encoded values).

### 5. Automated API Test
Run the automated test script:

```
python automated_api_test.py
```

---
**Note:**
- Use the encoding mappings from step 2 for categorical fields in predictions and API requests.
- The model uses: Age, Gender, Condition, Procedure, Cost, Length_of_Stay as features.