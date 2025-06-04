import google.generativeai as genai
import os
from typing import Dict, List
from time import sleep



genai.configure(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyC2OHi_8f0J8nFvWUF0hFb81MwgD5A4xe0"))
model = genai.GenerativeModel('gemini-1.5-flash')


def get_ai_suggestions(employee_data: Dict) -> List[str]:    
    
    DEFAULT_SUGGESTIONS = [
        "1. Conduct one-on-one meeting to understand concerns",
        "2. Provide career development opportunities",
        "3. Review workload and work-life balance"
    ]
    
    if model is None:
        return DEFAULT_SUGGESTIONS

    prompt = f"""
    You are an HR expert analyzing employee attrition risk. Based on the following employee data, provide 3 specific, actionable retention strategies.
    
    Employee Profile:
    - Attrition Risk Score: {employee_data['risk_score']:.2f}
    - Engagement Score: {employee_data['engagement_score']}/5
    - Satisfaction Score: {employee_data['satisfaction_score']}/5
    - Work-Life Balance: {employee_data['work_life_balance']}/5
    - Tenure: {employee_data['tenure']} years
    - Current Rating: {employee_data['current_rating']}/5
    - Performance Score: {employee_data['performance_score']}
    
    Provide exactly 3 numbered strategies that are:
    - Specific to this employee's profile
    - Actionable by HR/managers
    - Focused on the lowest scoring areas
    - Formatted as numbered list (1., 2., 3.)
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            
            if not response.text:
                raise ValueError("Empty response from model")
                
            
            suggestions = []
            for line in response.text.strip().split('\n'):
                line = line.strip()
                if line and line[0].isdigit():  
                    suggestions.append(line.split('.', 1)[1].strip() if '.' in line else line)
                if len(suggestions) >= 3:
                    break
            
            return suggestions[:3] or DEFAULT_SUGGESTIONS
            
        except Exception as e:
            if attempt == max_retries - 1:
                return DEFAULT_SUGGESTIONS
            sleep(1)  