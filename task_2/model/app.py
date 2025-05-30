from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from suggestions import get_ai_suggestions

app = FastAPI()

model = joblib.load('attrition_model.pkl')
encoder = joblib.load('performance_encoder.pkl')
scaler = joblib.load('scaler.pkl')

class EmployeeData(BaseModel):
    engagement_score: float
    satisfaction_score: float  
    work_life_balance: float
    tenure: float
    performance_score: str
    current_rating: int

@app.post("/predict")
def predict(data: EmployeeData):
    try:
        perf_encoded = encoder.transform([data.performance_score])[0]
        
        numeric_features = np.array([[
            data.engagement_score,
            data.satisfaction_score,
            data.work_life_balance,
            data.tenure,
            data.current_rating
        ]])
        
        numeric_scaled = scaler.transform(numeric_features)[0]
        
        input_data = np.append(numeric_scaled, perf_encoded).reshape(1, -1)
        
        input_df = pd.DataFrame(input_data, columns=[
            'Engagement Score', 'Satisfaction Score', 'Work-Life Balance Score',
            'Tenure', 'Current Employee Rating', 'Performance Score'
        ])
        
        probabilities = model.predict_proba(input_df)[0]
        risk_probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        
        employee_context = {
            'risk_score': risk_probability,
            'engagement_score': data.engagement_score,
            'satisfaction_score': data.satisfaction_score,
            'work_life_balance': data.work_life_balance,
            'tenure': data.tenure,
            'current_rating': data.current_rating,
            'performance_score': data.performance_score
        }
        
        suggestions = get_ai_suggestions(employee_context)
        
        return {
            "attrition_risk": round(risk_probability, 3),
            "risk_level": "High" if risk_probability > 0.7 else "Medium" if risk_probability > 0.4 else "Low",
            "suggestions": suggestions
        }
    
    except Exception as e:
        return {
            "error": f"Prediction failed: {str(e)}",
            "attrition_risk": 0.5,
            "risk_level": "Unknown",
            "suggestions": ["Error occurred during prediction"]
        }

@app.get("/")
def root():
    return {"message": "Employee Attrition Prediction API v2.0"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)