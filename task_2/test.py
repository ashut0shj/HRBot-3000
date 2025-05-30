import requests
import json

url = "http://localhost:8000/predict"

test_cases = [
    {
        "name": "High Risk Employee",
        "data": {
            "engagement_score": 1.5,
            "satisfaction_score": 2.0,
            "work_life_balance": 2.0,
            "tenure": 0.5,
            "performance_score": "Needs Improvement",
            "current_rating": 2
        }
    },
    {
        "name": "Medium Risk Employee",
        "data": {
            "engagement_score": 3.0,
            "satisfaction_score": 3.5,
            "work_life_balance": 3.0,
            "tenure": 2.0,
            "performance_score": "Fully Meets",
            "current_rating": 3
        }
    },
    {
        "name": "Low Risk Employee",
        "data": {
            "engagement_score": 4.5,
            "satisfaction_score": 4.5,
            "work_life_balance": 4.0,
            "tenure": 5.0,
            "performance_score": "Exceeds",
            "current_rating": 4
        }
    }
]

def test_api():
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        
        try:
            response = requests.post(url, json=test_case['data'])
            
            if response.status_code == 200:
                result = response.json()
                print(f"Risk Score: {result['attrition_risk']}")
                print(f"Risk Level: {result['risk_level']}")
                print(f"Suggestions: {result['suggestions']}")
            else:
                print(f"Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_api()