import subprocess
import time
import requests
import sys

# Start the FastAPI server in the background
server = subprocess.Popen([
    sys.executable, '-m', 'uvicorn', 'api:app', '--reload'
])

# Wait for the server to be ready
print("Waiting for API server to start...")
for _ in range(30):  # Try for up to 30 seconds
    try:
        r = requests.get("http://127.0.0.1:8000/docs")
        if r.status_code == 200:
            print("API server is ready.")
            break
    except requests.exceptions.ConnectionError:
        pass
    time.sleep(1)
else:
    print("API server did not start in time.")
    server.terminate()
    sys.exit(1)

# Example patient data for API testing
patient = {
    "age": 65,
    "num_comorbidities": 2,
    "lab_trend": 0.5,
    "discharge_home": 1
}

try:
    response = requests.post("http://127.0.0.1:8000/predict", json=patient)
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())
finally:
    # Stop the server
    server.terminate()
    print("API server stopped.")
