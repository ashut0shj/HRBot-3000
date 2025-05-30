from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from suggestions import get_ai_suggestions

app = FastAPI()

# Load models at startup
model = joblib.load('attrition_model.pkl')
encoder = joblib.load('performance_encoder.pkl')

class EmployeeData(BaseModel):
    engagement_score: float
    satisfaction_score: float  
    work_life_balance: float
    tenure: float
    performance_score: str
    current_rating: int

@app.post("/predict")
def predict(data: EmployeeData):
    # Convert performance score to numerical value
    perf_encoded = encoder.transform([data.performance_score])[0]
    
    # Load training data to get proper feature names
    X_train = pd.read_csv('data/X_train.csv')
    
    # Prepare input with proper feature names
    input_data = {
        X_train.columns[0]: data.engagement_score,
        X_train.columns[1]: data.satisfaction_score,
        X_train.columns[2]: data.work_life_balance,
        X_train.columns[3]: data.tenure,
        X_train.columns[4]: data.current_rating,
        X_train.columns[5]: perf_encoded
    }
    
    input_df = pd.DataFrame([input_data])
    
    # Get attrition risk probability
    probabilities = model.predict_proba(input_df)[0]
    
    # Handle case where model only predicts one class
    if len(probabilities) > 1:
        risk_probability = probabilities[1]
    else:
        risk_probability = probabilities[0]
    
    # Prepare context for AI suggestions
    employee_context = {
        'risk_score': risk_probability,
        'engagement_score': data.engagement_score,
        'satisfaction_score': data.satisfaction_score,
        'work_life_balance': data.work_life_balance,
        'tenure': data.tenure,
        'current_rating': data.current_rating,
        'performance_score': data.performance_score
    }
    
    # Get AI-powered suggestions
    suggestions = get_ai_suggestions(employee_context)
    
    return {
        "attrition_risk": round(risk_probability, 3),
        "risk_level": get_risk_level(risk_probability),
        "suggestions": suggestions
    }

def get_risk_level(risk_score):
    if risk_score > 0.7:
        return "High"
    elif risk_score > 0.4:
        return "Medium"
    else:
        return "Low"

@app.get("/")
def root():
    return {"message": "Employee Attrition Prediction API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)