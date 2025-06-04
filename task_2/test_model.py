import requests
import pandas as pd
import csv

# Define API endpoint
url = "https://risk-model-app-12345.blueriver-19dcaa87.eastus2.azurecontainerapps.io/predict"

# Load input CSV
input_csv = "input/employee_input.csv"
output_csv = "output/employee_output.csv"

# Read input data
df = pd.read_csv(input_csv)

# Prepare output file
output_data = []

# Send each row to the API
for _, row in df.iterrows():
    payload = {
        "engagement_score": row["engagement_score"],
        "satisfaction_score": row["satisfaction_score"],
        "work_life_balance": row["work_life_balance"],
        "tenure": row["tenure"],
        "performance_score": row["performance_score"],
        "current_rating": row["current_rating"]
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            output_data.append({
                "id": row["id"],
                "name": row["name"],
                "attrition_risk": result["attrition_risk"],
                "risk_level": result["risk_level"],
                "suggestions": result["suggestions"]
            })
        else:
            output_data.append({
                "id": row["id"],
                "name": row["name"],
                "attrition_risk": "Error",
                "risk_level": f"{response.status_code}",
                "suggestions": response.text
            })
    except Exception as e:
        output_data.append({
            "id": row["id"],
            "name": row["name"],
            "attrition_risk": "Exception",
            "risk_level": "Error",
            "suggestions": str(e)
        })

# Save to output CSV
output_df = pd.DataFrame(output_data)
output_df.to_csv(output_csv, index=False)

print(f"Results written to {output_csv}")
