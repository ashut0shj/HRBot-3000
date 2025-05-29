import pdfplumber
import re
import json
from dotenv import load_dotenv
import os
import google.generativeai as genai
from typing import Dict, List, Union


load_dotenv()  
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


JD_REQUIREMENTS = {
    "required_skills": ["Python", "AWS", "Machine Learning", "TensorFlow", "SQL", "Docker"],
    "min_experience": 1,
    "required_degree": "Bachelor's in Computer Science",
    "preferred_locations": [""],
    "preferred_certs": [""]
}


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                
                text += page.extract_text(x_tolerance=2, y_tolerance=2) + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""
    
    
    text = re.sub(r'\n{2,}', '\n', text)  
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  
    return text.strip()


def extract_resume_data(resume_text: str) -> Dict[str, Union[List, float, str]]:

    prompt = f"""
    Analyze this resume and return EXACTLY the following JSON structure:
    {{
        "technical_skills": ["list", "of", "technical", "skills"],
        "total_experience": years (float),
        "role_experience": [{{"title": "job_title", "years": years, "organisation": "organisation_name"}}],
        "education": [{{"degree": "degree_name", "institution": "school"}}],
        "projects": [{{"name": "project_name", "tools": ["list"]}}],
        "certifications": ["list"],
        "location": "city/state/country"
    }}

    Rules:
    1. For skills: Include only hard technical skills (no soft skills)
    2. For experience: Calculate total years across all roles
    3. If location is not found, set it to "Unknown"
    4. Be strict with field names and JSON formatting

    Resume Text:
    {resume_text[:3000]}  
    """
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        
        
        json_str = response.text.replace("```json", "").replace("```", "").strip()
        print(f"Gemini response:\n{json_str}\n")
        data = json.loads(json_str)
        
        
        return {
            "technical_skills": data.get("technical_skills", []),
            "total_experience": data.get("total_experience", 0.0),
            "role_experience": data.get("role_experience", []),
            "education": data.get("education", []),
            "projects": data.get("projects", []),
            "certifications": data.get("certifications", []),
            "location": data.get("location", "Unknown")  
        }
    except Exception as e:
        print(f"Gemini processing error: {e}")
        return {
            "technical_skills": [],
            "total_experience": 0.0,
            "role_experience": [],
            "education": [],
            "projects": [],
            "certifications": [],
            "location": "Unknown"  
        }


def calculate_match_score(resume_data: dict, jd: dict) -> float:

    if not resume_data:
        return 0.0
    
    score = 0.0
    
    
    resume_skills = set(s.lower() for s in resume_data.get("technical_skills", []))
    jd_skills = set(s.lower() for s in jd["required_skills"])
    matched_skills = resume_skills.intersection(jd_skills)
    score += 0.4 * (len(matched_skills) / len(jd_skills) if jd_skills else 0)
    
    
    exp = resume_data.get("total_experience", 0)
    score += 0.25 if exp >= jd["min_experience"] else 0.25 * (exp / jd["min_experience"])
    
    
    resume_degree = resume_data.get("education", [{}])[0].get("degree", "").lower()
    required_degree = jd["required_degree"].lower()
    score += 0.15 if required_degree in resume_degree else 0
    
    
    project_score = 0
    for project in resume_data.get("projects", []):
        if any(tool.lower() in jd_skills for tool in project.get("tools", [])):
            project_score += 0.02  
    score += min(project_score, 0.10)
    
    
    resume_loc = resume_data.get("location", "Unknown").lower()  
    if resume_loc != "unknown":  
        score += 0.05 if any(loc.lower() in resume_loc for loc in jd["preferred_locations"]) else 0
    
    
    resume_certs = set(c.lower() for c in resume_data.get("certifications", []))
    score += 0.05 if any(c.lower() in resume_certs for c in jd["preferred_certs"]) else 0
    
    return round(score * 100, 1)


def process_resume(pdf_path: str) -> dict:
    
    resume_text = extract_text_from_pdf(pdf_path)
    
    if not resume_text:
        print("Failed to extract text from PDF")
        return {
            "error": "Failed to extract text from PDF",
            "match_score": 0,
            "resume_data": {}
        }
    
    
    print("Processing resume with Gemini...")
    resume_data = extract_resume_data(resume_text)
    
    print("\nExtracted Resume Data:")
    print(json.dumps(resume_data, indent=2))
    
    
    match_score = calculate_match_score(resume_data, JD_REQUIREMENTS)
    
    
    print("\n=== Results ===")
    print(f"JD Match Score: {match_score}%")
    if resume_data["technical_skills"]:
        print(f"Matched Skills: {set(resume_data['technical_skills']) & set(JD_REQUIREMENTS['required_skills'])}")
    print(f"Experience: {resume_data.get('total_experience', 0)} years (JD requires {JD_REQUIREMENTS['min_experience']}+)")
    print(f"Location: {resume_data.get('location', 'Unknown')}")
    
    
    return {
        "match_score": match_score,
        "resume_data": resume_data,
        "matched_skills": list(set(resume_data['technical_skills']) & set(JD_REQUIREMENTS['required_skills'])),
        "jd_requirements": JD_REQUIREMENTS
    }


if __name__ == "__main__":
    
    pdf_path = "Resume_Ashutosh.pdf"  
    result = process_resume(pdf_path)
    
    if "error" not in result:
        print("\n=== Final Results ===")
        print(f"Match Score: {result['match_score']}%")
    else:
        print(f"Error: {result['error']}")