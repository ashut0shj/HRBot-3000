import os
import json
import PyPDF2
import google.generativeai as genai
from typing import Dict, List, Any
import re
import time
from difflib import SequenceMatcher

class ResumeScreeningTool:
    def __init__(self, gemini_api_key: str):
        """Initialize the Gemini AI client"""
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')  # Cost-effective model
        
        # Define token mappings for consistency
        self.skill_tokens = {
            # Technical Skills
            "python": ["python", "py"],
            "javascript": ["javascript", "js", "node.js", "nodejs"],
            "java": ["java", "spring", "springboot"],
            "react": ["react", "reactjs", "react.js"],
            "angular": ["angular", "angularjs"],
            "machine_learning": ["machine learning", "ml", "artificial intelligence", "ai", "deep learning", "neural networks"],
            "cloud": ["aws", "azure", "gcp", "google cloud", "cloud computing"],
            "database": ["sql", "mysql", "postgresql", "mongodb", "nosql"],
            "devops": ["docker", "kubernetes", "jenkins", "ci/cd", "devops"],
            
            # Soft Skills
            "leadership": ["leadership", "team lead", "mentor", "mentoring", "leading"],
            "communication": ["communication", "presentation", "public speaking", "interpersonal"],
            "problem_solving": ["problem solving", "analytical", "troubleshooting", "debugging"],
            "teamwork": ["teamwork", "collaboration", "team player", "cross-functional"],
            "project_management": ["project management", "agile", "scrum", "planning"],
            
            # Education
            "computer_science": ["computer science", "cs", "cse", "computer engineering"],
            "information_technology": ["information technology", "it", "information systems"],
            "engineering": ["engineering", "btech", "be", "bachelor of technology"],
            "masters": ["masters", "mtech", "ms", "master of science", "mba"],
            "phd": ["phd", "doctorate", "doctoral"]
        }

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            print(f"Error extracting PDF: {e}")
            return ""

    def get_completion(self, prompt: str, max_retries: int = 3) -> str:
        """Get completion from Gemini API with retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=2048,
                    )
                )
                return response.text
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                else:
                    print(f"All attempts failed: {e}")
                    return ""

    def calculate_similarity_score(self, jd_data: Dict[str, Any], resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate similarity score between JD and Resume using direct comparison"""
        
        # Extract skills from JD
        jd_tech_skills = set([skill.lower() for skill in jd_data.get('technical_skills', [])])
        jd_soft_skills = set([skill.lower() for skill in jd_data.get('soft_skills', [])])
        jd_education = set([edu.lower() for edu in jd_data.get('education', [])])
        
        # Extract skills from Resume
        resume_tech_skills = set()
        resume_soft_skills = set()
        resume_education = set()
        
        # Handle different resume formats
        if isinstance(resume_data.get('technical_skills'), list):
            for skill in resume_data.get('technical_skills', []):
                if isinstance(skill, dict):
                    resume_tech_skills.add(skill.get('skill', '').lower())
                else:
                    resume_tech_skills.add(str(skill).lower())
        
        if isinstance(resume_data.get('soft_skills'), list):
            for skill in resume_data.get('soft_skills', []):
                if isinstance(skill, dict):
                    resume_soft_skills.add(skill.get('skill', '').lower())
                else:
                    resume_soft_skills.add(str(skill).lower())
                    
        if isinstance(resume_data.get('education'), list):
            for edu in resume_data.get('education', []):
                if isinstance(edu, dict):
                    resume_education.add(edu.get('degree', '').lower())
                else:
                    resume_education.add(str(edu).lower())
        
        # Calculate matches and misses
        tech_matches = jd_tech_skills.intersection(resume_tech_skills)
        tech_misses = jd_tech_skills - resume_tech_skills
        
        soft_matches = jd_soft_skills.intersection(resume_soft_skills)
        soft_misses = jd_soft_skills - resume_soft_skills
        
        edu_matches = jd_education.intersection(resume_education)
        edu_misses = jd_education - resume_education
        
        # Calculate percentages
        tech_match_percent = (len(tech_matches) / len(jd_tech_skills)) * 100 if jd_tech_skills else 0
        soft_match_percent = (len(soft_matches) / len(jd_soft_skills)) * 100 if jd_soft_skills else 0
        edu_match_percent = (len(edu_matches) / len(jd_education)) * 100 if jd_education else 0
        
        # Overall similarity (weighted average)
        overall_similarity = (
            tech_match_percent * 0.5 +  # 50% weight on technical skills
            soft_match_percent * 0.3 +  # 30% weight on soft skills
            edu_match_percent * 0.2     # 20% weight on education
        )
        
        return {
            "overall_similarity": round(overall_similarity, 2),
            "technical_skills": {
                "match_percentage": round(tech_match_percent, 2),
                "matches": list(tech_matches),
                "missing": list(tech_misses)
            },
            "soft_skills": {
                "match_percentage": round(soft_match_percent, 2),
                "matches": list(soft_matches),
                "missing": list(soft_misses)
            },
            "education": {
                "match_percentage": round(edu_match_percent, 2),
                "matches": list(edu_matches),
                "missing": list(edu_misses)
            }
        }

    def process_job_description(self, jd_text: str) -> Dict[str, Any]:
        """Convert job description to structured JSON with tokens"""
        
        prompt = f"""
        Analyze this job description and convert it to JSON format with the following structure:
        {{
            "role": "job title",
            "experience_required": "min specific requirement",
            "technical_skills": ["skill1", "skill2", ...],
            "soft_skills": ["skill1", "skill2", ...],
            "education": ["degree requirements"],
            "responsibilities": ["key responsibilities"],
            "nice_to_have": ["optional skills/experience"],
            "company_info": "brief company context"
        }}

        Use these skill tokens for consistency:
        Technical: python, javascript, java, react, angular, machine_learning, cloud, database, devops
        Soft Skills: leadership, communication, problem_solving, teamwork, project_management
        Education: computer_science, information_technology, engineering, masters, phd

        Job Description:
        {jd_text}

        Return only valid JSON without any additional text or formatting:
        """
        
        response = self.get_completion(prompt)
        try:
            # Clean the response to extract JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                return json.loads(response)
        except Exception as e:
            print(f"Error parsing JD JSON: {e}")
            return self._create_fallback_jd()

    def process_resume(self, resume_text: str, jd_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Convert resume to structured JSON with scoring"""
        
        jd_example = json.dumps(jd_structure, indent=2)
        
        prompt = f"""
        Analyze this resume and convert it to JSON format matching the job description structure.
        Also provide scoring based on relevance and achievements.

        Expected JSON structure (match the JD format):
        {jd_example}

        For the resume, create JSON with these additional scoring fields:
        {{
            "role": "current/target role",
            "total_experience": "X years",
            "technical_skills": [
                {{"skill": "skill_name", "score": 1-10, "evidence": "where mentioned/used"}}
            ],
            "soft_skills": [
                {{"skill": "skill_name", "score": 1-10, "evidence": "examples from experience"}}
            ],
            "education": [
                {{"degree": "degree_name", "specialization": "field", "score": 1-10}}
            ],
            "work_experience": [
                {{"company": "name", "role": "position", "duration": "time", "tech_relevance": 1-10}}
            ],
            "projects": [
                {{"name": "project_name", "technologies": ["tech1", "tech2"], "score": 1-10}}
            ],
            "achievements": [
                {{"type": "hackathon/leetcode/certification/etc", "description": "details", "bonus_score": 1-5}}
            ],
            "resume_quality_scores": {{
                "technical_strength": 0-100,
                "soft_skills_strength": 0-100,
                "experience_quality": 0-100,
                "education_quality": 0-100,
                "bonus_points": 0-20
            }}
        }}

        Scoring Guidelines:
        - Technical Skills: 10 = Expert level with project evidence, 5 = Mentioned/Basic, 1 = Barely relevant
        - Soft Skills: Extract from experience descriptions, leadership roles, team projects
        - Experience: Higher score for relevant tech companies, senior roles, long tenure
        - Achievements: Bonus points for hackathons, competitive programming, certifications
        - Resume Quality Scores: Rate the overall quality/strength of each section (0-100)

        Resume Text:
        {resume_text}

        Return only valid JSON without any additional text or formatting:
        """
        
        response = self.get_completion(prompt)
        try:
            # Clean the response to extract JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                return json.loads(response)
        except Exception as e:
            print(f"Error parsing resume JSON: {e}")
            return self._create_fallback_resume()

    def calculate_final_score(self, jd_data: Dict[str, Any], resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate final matching score and detailed analysis"""
        
        # Extract scores from resume data
        resume_quality_scores = resume_data.get('resume_quality_scores', {})
        
        technical_strength = resume_quality_scores.get('technical_strength', 0)
        soft_skills_strength = resume_quality_scores.get('soft_skills_strength', 0)
        experience_quality = resume_quality_scores.get('experience_quality', 0)
        education_quality = resume_quality_scores.get('education_quality', 0)
        bonus_points = resume_quality_scores.get('bonus_points', 0)
        
        # Calculate JD-Resume similarity using code-based approach
        similarity_data = self.calculate_similarity_score(jd_data, resume_data)
        
        # Use similarity scores for JD matching
        tech_jd_match = similarity_data['technical_skills']['match_percentage']
        soft_jd_match = similarity_data['soft_skills']['match_percentage']
        edu_jd_match = similarity_data['education']['match_percentage']
        
        # Calculate weighted final score using JD matches and resume quality
        final_score = (
            tech_jd_match * 0.3 +        # 30% weight on JD technical match
            technical_strength * 0.2 +    # 20% weight on technical skill strength
            experience_quality * 0.2 +    # 20% weight on experience quality
            soft_jd_match * 0.15 +       # 15% weight on JD soft skills match
            soft_skills_strength * 0.1 +  # 10% weight on soft skills strength
            edu_jd_match * 0.05 +        # 5% weight on JD education match
            bonus_points * 0.5           # Bonus points (multiplied for impact)
        )
        
        # Cap at 100%
        final_score = min(final_score, 100)
        
        # Generate recommendations
        recommendations = []
        if tech_jd_match < 70:
            recommendations.append("Strengthen technical skills mentioned in JD")
        if experience_quality < 60:
            recommendations.append("Gain more relevant industry experience")
        if soft_jd_match < 70:
            recommendations.append("Highlight leadership and communication experiences")
        
        return {
            "final_score": round(final_score, 2),
            "jd_matching_scores": {
                "technical_skills_match": tech_jd_match,
                "soft_skills_match": soft_jd_match,
                "education_match": edu_jd_match
            },
            "resume_quality_scores": {
                "technical_strength": technical_strength,
                "soft_skills_strength": soft_skills_strength,
                "experience_quality": experience_quality,
                "education_quality": education_quality,
                "bonus_points": bonus_points
            },
            "recommendation": "HIRE" if final_score >= 75 else "MAYBE" if final_score >= 60 else "REJECT",
            "strengths": self._identify_strengths(resume_data),
            "gaps": self._identify_gaps_fixed(jd_data, resume_data),
            "recommendations": recommendations
        }

    def _identify_strengths(self, resume_data: Dict[str, Any]) -> List[str]:
        """Identify candidate strengths"""
        strengths = []
        
        # Check technical skills with high scores
        for skill in resume_data.get('technical_skills', []):
            if isinstance(skill, dict) and skill.get('score', 0) >= 8:
                strengths.append(f"Strong in {skill.get('skill', 'Unknown')}")
        
        # Check achievements
        achievements = resume_data.get('achievements', [])
        if achievements:
            strengths.append(f"Notable achievements: {len(achievements)} items")
        
        # Check experience
        if resume_data.get('total_experience', '0').replace(' years', '').replace('+', '').isdigit():
            exp_years = int(resume_data.get('total_experience', '0').replace(' years', '').replace('+', ''))
            if exp_years >= 5:
                strengths.append(f"Experienced professional ({exp_years}+ years)")
        
        return strengths

    def _identify_gaps_fixed(self, jd_data: Dict[str, Any], resume_data: Dict[str, Any]) -> List[str]:
        """Identify skill gaps - FIXED VERSION"""
        gaps = []
        
        # Get JD technical skills
        jd_tech_skills = set([skill.lower() for skill in jd_data.get('technical_skills', [])])
        
        # Get resume technical skills (handle both dict and string formats)
        resume_tech_skills = set()
        for skill in resume_data.get('technical_skills', []):
            if isinstance(skill, dict):
                resume_tech_skills.add(skill.get('skill', '').lower())
            else:
                resume_tech_skills.add(str(skill).lower())
        
        # Find missing skills
        missing_skills = jd_tech_skills - resume_tech_skills
        for skill in missing_skills:
            gaps.append(f"Missing: {skill}")
        
        return gaps

    def _create_fallback_jd(self) -> Dict[str, Any]:
        """Fallback JD structure"""
        return {
            "role": "Software Engineer",
            "experience_required": "2-5 years",
            "technical_skills": ["python", "javascript", "database"],
            "soft_skills": ["communication", "teamwork"],
            "education": ["computer_science"],
            "responsibilities": ["Software development"],
            "nice_to_have": [],
            "company_info": "Tech company"
        }

    def _create_fallback_resume(self) -> Dict[str, Any]:
        """Fallback resume structure"""
        return {
            "role": "Software Developer",
            "total_experience": "0 years",
            "technical_skills": [],
            "soft_skills": [],
            "education": [],
            "work_experience": [],
            "projects": [],
            "achievements": [],
            "resume_quality_scores": {
                "technical_strength": 0,
                "soft_skills_strength": 0,
                "experience_quality": 0,
                "education_quality": 0,
                "bonus_points": 0
            }
        }

    def print_json_files(self, jd_data: Dict[str, Any], resume_data: Dict[str, Any]):
        """Print both JSON files in a formatted way"""
        print("\n" + "="*60)
        print("📄 JOB DESCRIPTION JSON")
        print("="*60)
        print(json.dumps(jd_data, indent=2))
        
        print("\n" + "="*60)
        print("📄 RESUME JSON")
        print("="*60)
        print(json.dumps(resume_data, indent=2))

    def screen_resume(self, jd_text: str, resume_pdf_path: str) -> Dict[str, Any]:
        """Main function to screen a resume against job description"""
        
        print("🔄 Processing job description...")
        jd_data = self.process_job_description(jd_text)
        
        print("🔄 Extracting resume text...")
        resume_text = self.extract_text_from_pdf(resume_pdf_path)
        
        if not resume_text:
            return {"error": "Could not extract text from PDF"}
        
        print("🔄 Processing resume...")
        resume_data = self.process_resume(resume_text, jd_data)
        
        print("🔄 Calculating final score...")
        final_results = self.calculate_final_score(jd_data, resume_data)
        
        print("🔄 Calculating similarity score...")
        similarity_results = self.calculate_similarity_score(jd_data, resume_data)
        
        # Print JSON files
        self.print_json_files(jd_data, resume_data)
        
        return {
            "job_description": jd_data,
            "resume_analysis": resume_data,
            "screening_results": final_results,
            "similarity_analysis": similarity_results,
            "timestamp": "2025-06-03"
        }

    def batch_screen_resumes(self, jd_text: str, resume_folder: str) -> List[Dict[str, Any]]:
        """Screen multiple resumes in a folder"""
        results = []
        
        for filename in os.listdir(resume_folder):
            if filename.endswith('.pdf'):
                print(f"📄 Processing {filename}...")
                try:
                    result = self.screen_resume(jd_text, os.path.join(resume_folder, filename))
                    result['filename'] = filename
                    results.append(result)
                except Exception as e:
                    print(f"❌ Error processing {filename}: {e}")
                    results.append({
                        'filename': filename,
                        'error': str(e),
                        'screening_results': {'final_score': 0, 'recommendation': 'ERROR'}
                    })
        
        # Sort by score
        results.sort(key=lambda x: x.get('screening_results', {}).get('final_score', 0), reverse=True)
        return results


# Usage Example with Gemini API
def main():
    # Initialize the tool with Gemini API key
    tool = ResumeScreeningTool(
        gemini_api_key="AIzaSyC2OHi_8f0J8nFvWUF0hFb81MwgD5A4xe0"  # Get from Google AI Studio
    )
    
    # Sample job description (LinkedIn format)
    job_description = """
    Software Engineer - Full Stack Development
    
    Company: TechCorp Solutions
    Location: Bangalore, India
    Experience: 3-5 years
    
    Job Description:
    We are looking for a skilled Software Engineer to join our dynamic team. 
    
    Responsibilities:
    - Design and develop scalable web applications using React and Node.js
    - Work with databases (PostgreSQL, MongoDB)
    - Collaborate with cross-functional teams
    - Implement CI/CD pipelines
    - Mentor junior developers
    
    Required Skills:
    - 3+ years of experience in full-stack development
    - Strong proficiency in JavaScript, Python
    - Experience with React, Node.js
    - Database design and optimization
    - Problem-solving and analytical skills
    - Excellent communication skills
    
    Preferred Qualifications:
    - Bachelor's degree in Computer Science or related field
    - Experience with cloud platforms (AWS/Azure)
    - Knowledge of machine learning concepts
    - Previous leadership experience
    
    What We Offer:
    - Competitive salary
    - Health insurance
    - Learning and development opportunities
    - Flexible work arrangements
    """
    
    # Screen a single resume
    try:
        results = tool.screen_resume(job_description, "Resume_Ashutosh.pdf")
        
        print("\n" + "="*50)
        print("📊 RESUME SCREENING RESULTS")
        print("="*50)
        
        screening = results["screening_results"]
        print(f"📈 Final Score: {screening['final_score']}%")
        print(f"🏷️  Recommendation: {screening['recommendation']}")
        
        print(f"\n📋 JD Matching Scores:")
        for category, score in screening['jd_matching_scores'].items():
            print(f"   {category.replace('_', ' ').title()}: {score}%")
        
        print(f"\n📋 Resume Quality Scores:")
        for category, score in screening['resume_quality_scores'].items():
            print(f"   {category.replace('_', ' ').title()}: {score}%")
        
        print(f"\n💪 Strengths:")
        for strength in screening['strengths']:
            print(f"   ✓ {strength}")
        
        print(f"\n🔍 Gaps:")
        for gap in screening['gaps']:
            print(f"   ⚠️ {gap}")
        
        print(f"\n💡 Recommendations:")
        for rec in screening['recommendations']:
            print(f"   → {rec}")
        
        # Print similarity analysis
        similarity = results["similarity_analysis"]
        print(f"\n🎯 SIMILARITY ANALYSIS (Code-based)")
        print("="*50)
        print(f"📊 Overall Similarity: {similarity['overall_similarity']}%")
        
        print(f"\n🔧 Technical Skills:")
        print(f"   Match: {similarity['technical_skills']['match_percentage']}%")
        print(f"   Found: {', '.join(similarity['technical_skills']['matches']) if similarity['technical_skills']['matches'] else 'None'}")
        print(f"   Missing: {', '.join(similarity['technical_skills']['missing']) if similarity['technical_skills']['missing'] else 'None'}")
        
        print(f"\n🤝 Soft Skills:")
        print(f"   Match: {similarity['soft_skills']['match_percentage']}%")
        print(f"   Found: {', '.join(similarity['soft_skills']['matches']) if similarity['soft_skills']['matches'] else 'None'}")
        print(f"   Missing: {', '.join(similarity['soft_skills']['missing']) if similarity['soft_skills']['missing'] else 'None'}")
        
        print(f"\n🎓 Education:")
        print(f"   Match: {similarity['education']['match_percentage']}%")
        print(f"   Found: {', '.join(similarity['education']['matches']) if similarity['education']['matches'] else 'None'}")
        print(f"   Missing: {', '.join(similarity['education']['missing']) if similarity['education']['missing'] else 'None'}")
            
        # Save detailed results
        with open("screening_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Detailed results saved to screening_results.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")

    # Example: Batch processing
    try:
        print("\n🔄 Starting batch processing...")
        batch_results = tool.batch_screen_resumes(job_description, "resumes_folder/")
        
        print("\n📊 BATCH SCREENING RESULTS")
        print("="*50)
        
        for i, result in enumerate(batch_results[:5], 1):  # Show top 5
            if 'error' not in result:
                score = result['screening_results']['final_score']
                recommendation = result['screening_results']['recommendation']
                similarity = result.get('similarity_analysis', {}).get('overall_similarity', 0)
                print(f"{i}. {result['filename']}: {score}% (AI) | {similarity}% (Code) | ({recommendation})")
            else:
                print(f"{i}. {result['filename']}: ERROR - {result['error']}")
        
        # Save batch results
        with open("batch_screening_results.json", "w") as f:
            json.dump(batch_results, f, indent=2)
        print(f"\n💾 Batch results saved to batch_screening_results.json")
        
    except Exception as e:
        print(f"❌ Batch processing error: {e}")


if __name__ == "__main__":
    main()