import requests

# Example patient data for API testing
patient = {
    "age": 65,
    "num_comorbidities": 2,
    "lab_trend": 0.5,
    "discharge_home": 1
}

response = requests.post("http://127.0.0.1:8000/predict", json=patient)
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
