from flask import Flask, render_template, request
import requests

app = Flask(__name__)

API_URL = "https://risk-model-app-12345.blueriver-19dcaa87.eastus2.azurecontainerapps.io/predict"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            "engagement_score": float(request.form['engagement_score']),
            "satisfaction_score": float(request.form['satisfaction_score']),
            "work_life_balance": float(request.form['work_life_balance']),
            "tenure": float(request.form['tenure']),
            "performance_score": request.form['performance_score'],
            "current_rating": int(request.form['current_rating'])
        }
        
        response = requests.post(API_URL, json=data)
        
        if response.status_code == 200:
            result = response.json()
            return render_template('result.html', 
                                 risk_score=result['attrition_risk'],
                                 risk_level=result['risk_level'],
                                 suggestions=result['suggestions'])
        else:
            return render_template('result.html', error=f"Error: {response.status_code}")
            
    except Exception as e:
        return render_template('result.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)