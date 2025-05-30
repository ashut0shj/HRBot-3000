import google.generativeai as genai
import os
from typing import Dict, List

genai.configure(api_key="AIzaSyC2OHi_8f0J8nFvWUF0hFb81MwgD5A4xe0")
model = genai.GenerativeModel('gemini-1.5-flash')

def get_ai_suggestions(employee_data: Dict) -> List[str]:
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
    
    Provide exactly 3 numbered strategies that are specific, actionable, and focused on the lowest scoring areas.
    Format: Return only the 3 numbered strategies, nothing else.
    """
    
    try:
        response = model.generate_content(prompt)
        suggestions_text = response.text.strip()
        
        suggestions = []
        for line in suggestions_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('•') or line.startswith('-')):
                suggestions.append(line)
        
        return suggestions[:3] if len(suggestions) >= 3 else ["Error: Could not generate suggestions"]
        
    except Exception as e:
        return [f"Error: {str(e)}"]