import google.generativeai as genai
import os
from typing import Dict, List

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')

def get_ai_suggestions(employee_data: Dict) -> List[str]:
    """
    Generate retention strategies using Gemini AI based on employee data
    """
    
    # Create prompt for Gemini
    prompt = f"""
    You are an HR expert analyzing employee attrition risk. Based on the following employee data, provide 3 specific, actionable retention strategies.
    
    Employee Profile:
    - Attrition Risk Score: {employee_data['risk_score']:.2f} (0.0 = no risk, 1.0 = high risk)
    - Engagement Score: {employee_data['engagement_score']}/5
    - Satisfaction Score: {employee_data['satisfaction_score']}/5
    - Work-Life Balance: {employee_data['work_life_balance']}/5
    - Tenure: {employee_data['tenure']} years
    - Current Rating: {employee_data['current_rating']}/5
    - Performance Score: {employee_data['performance_score']}
    
    Provide exactly 3 numbered strategies that are:
    1. Specific to this employee's situation
    2. Actionable by managers/HR
    3. Focused on the lowest scoring areas
    
    Format: Return only the 3 numbered strategies, nothing else.
    """
    
    try:
        response = model.generate_content(prompt)
        suggestions_text = response.text.strip()
        
        # Parse the response into a list
        suggestions = []
        for line in suggestions_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('•') or line.startswith('-')):
                suggestions.append(line)
        
        # Ensure we have exactly 3 suggestions
        if len(suggestions) < 3:
            suggestions.extend(get_fallback_suggestions(employee_data['risk_score']))
        
        return suggestions[:3]
        
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return get_fallback_suggestions(employee_data['risk_score'])

def get_fallback_suggestions(risk_score: float) -> List[str]:
    """
    Fallback suggestions if API fails
    """
    if risk_score > 0.7:
        return ['frrr',
            "1. Schedule immediate one-on-one meeting to discuss concerns and career goals",
            "2. Review current compensation and benefits package for competitive alignment",
            "3. Implement flexible work arrangements or additional time-off options"
        ]
    elif risk_score > 0.4:
        return ['fff',
            "1. Increase frequency of check-ins and provide regular feedback",
            "2. Explore professional development and training opportunities",
            "3. Assign challenging projects to improve engagement and satisfaction"
        ]
    else:
        return ['rrr',
            "1. Continue current management approach and recognize good performance",
            "2. Consider for leadership development or mentoring opportunities",
            "3. Maintain regular communication and gather feedback on job satisfaction"
        ]

def create_suggestion_prompt(employee_data: Dict) -> str:
    """
    Create a formatted prompt that can be used with any LLM
    """
    return f"""
    Analyze this employee's attrition risk and provide retention strategies:
    
    Risk Score: {employee_data['risk_score']:.2f}
    Engagement: {employee_data['engagement_score']}/5
    Satisfaction: {employee_data['satisfaction_score']}/5
    Work-Life Balance: {employee_data['work_life_balance']}/5
    Tenure: {employee_data['tenure']} years
    Performance Rating: {employee_data['current_rating']}/5
    Performance Level: {employee_data['performance_score']}
    
    Provide 3 specific retention strategies for this employee.
    """