import os
import json
import csv
import PyPDF2
import google.generativeai as genai
from typing import Dict, List, Any
import re
import time
from difflib import SequenceMatcher


class ResumeScreeningTool:
    def __init__(self, api_key, model='gemini-1.5-flash'):
        self.api_key = api_key
        self.model_v = model
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_v)
        self.jd_cache = {}
        
        self.skill_tokens = {
            "python": ["python", "py"],
            "javascript": ["javascript", "js", "node.js", "nodejs"],
            "java": ["java", "spring", "springboot"],
            "react": ["react", "reactjs", "react.js"],
            "angular": ["angular", "angularjs"],
            "machine_learning": ["machine learning", "ml", "artificial intelligence", "ai", "deep learning", "neural networks"],
            "cloud": ["aws", "azure", "gcp", "google cloud", "cloud computing"],
            "database": ["sql", "mysql", "postgresql", "mongodb", "nosql"],
            "devops": ["docker", "kubernetes", "jenkins", "ci/cd", "devops"],
            "leadership": ["leadership", "team lead", "mentor", "mentoring", "leading"],
            "communication": ["communication", "presentation", "public speaking", "interpersonal"],
            "problem_solving": ["problem solving", "analytical", "troubleshooting", "debugging"],
            "teamwork": ["teamwork", "collaboration", "team player", "cross-functional"],
            "project_management": ["project management", "agile", "scrum", "planning"],
            "computer_science": ["computer science", "cs", "cse", "computer engineering"],
            "information_technology": ["information technology", "it", "information systems"],
            "engineering": ["engineering", "btech", "be", "bachelor of technology"],
            "masters": ["masters", "mtech", "ms", "master of science", "mba"],
            "phd": ["phd", "doctorate", "doctoral"]
        }

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                txt = ""
                for page in reader.pages:
                    txt += page.extract_text()
                return txt
        except Exception as e:
            raise Exception(f"Error extracting PDF: {e}")

    def load_jd_from_file(self, jd_file_path: str) -> str:
        """Load job description from text file"""
        try:
            with open(jd_file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            raise Exception(f"Error reading JD file: {e}")

    def get_completion(self, prompt: str, max_retries: int = 3) -> str:
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
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    raise Exception(f"All attempts failed: {e}")

    def calc_similarity_score(self, jd: Dict[str, Any], res: Dict[str, Any]) -> Dict[str, Any]:
        jd_tech = set([skill.lower() for skill in jd.get('technical_skills', [])])
        jd_soft = set([skill.lower() for skill in jd.get('soft_skills', [])])
        jd_edu = set([edu.lower() for edu in jd.get('education', [])])
        
        res_tech = set()
        res_soft = set()
        res_edu = set()
        
        if isinstance(res.get('technical_skills'), list):
            for skill in res.get('technical_skills', []):
                if isinstance(skill, dict):
                    res_tech.add(skill.get('skill', '').lower())
                else:
                    res_tech.add(str(skill).lower())
        
        if isinstance(res.get('soft_skills'), list):
            for skill in res.get('soft_skills', []):
                if isinstance(skill, dict):
                    res_soft.add(skill.get('skill', '').lower())
                else:
                    res_soft.add(str(skill).lower())
                    
        if isinstance(res.get('education'), list):
            for edu in res.get('education', []):
                if isinstance(edu, dict):
                    res_edu.add(edu.get('degree', '').lower())
                else:
                    res_edu.add(str(edu).lower())
        
        tech_match = jd_tech.intersection(res_tech)
        tech_miss = jd_tech - res_tech
        
        soft_match = jd_soft.intersection(res_soft)
        soft_miss = jd_soft - res_soft
        
        edu_match = jd_edu.intersection(res_edu)
        edu_miss = jd_edu - res_edu
        
        tech_pct = (len(tech_match) / len(jd_tech)) * 100 if jd_tech else 0
        soft_pct = (len(soft_match) / len(jd_soft)) * 100 if jd_soft else 0
        edu_pct = (len(edu_match) / len(jd_edu)) * 100 if jd_edu else 0
        
        overall = (tech_pct * 0.5 + soft_pct * 0.3 + edu_pct * 0.2)
        
        return {
            "overall_similarity": round(overall, 2),
            "technical_skills": {
                "match_percentage": round(tech_pct, 2),
                "matches": list(tech_match),
                "missing": list(tech_miss)
            },
            "soft_skills": {
                "match_percentage": round(soft_pct, 2),
                "matches": list(soft_match),
                "missing": list(soft_miss)
            },
            "education": {
                "match_percentage": round(edu_pct, 2),
                "matches": list(edu_match),
                "missing": list(edu_miss)
            }
        }

    def process_jd_cached(self, jd_txt: str) -> Dict[str, Any]:
        jd_hash = hash(jd_txt)
        
        if jd_hash in self.jd_cache:
            return self.jd_cache[jd_hash]
        
        jd_data = self.process_jd(jd_txt)
        self.jd_cache[jd_hash] = jd_data
        return jd_data

    def process_jd(self, jd_txt: str) -> Dict[str, Any]:
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
        {jd_txt}

        Return only valid JSON without any additional text or formatting:
        """
        
        resp = self.get_completion(prompt)
        try:
            json_start = resp.find('{')
            json_end = resp.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = resp[json_start:json_end]
                return json.loads(json_str)
            else:
                return json.loads(resp)
        except Exception as e:
            raise Exception(f"Error parsing JD JSON: {e}")

    def process_resume(self, res_txt: str, jd_struct: Dict[str, Any]) -> Dict[str, Any]:
        jd_ex = json.dumps(jd_struct, indent=2)
        
        prompt = f"""
        Analyze this resume and convert it to JSON format matching the job description structure.
        Also provide scoring based on relevance and achievements.

        Expected JSON structure (match the JD format):
        {jd_ex}

        For the resume, create JSON with these additional scoring fields:
        {{
            "name": "candidate full name",
            "email": "candidate email address",
            "phone": "candidate phone number",
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

        IMPORTANT: Extract the candidate's name, email, and phone number from the resume text. Look for:
        - Name: Usually at the top of the resume or in header
        - Email: Look for @ symbol and email formats
        - Phone: Look for phone number patterns

        Scoring Guidelines:
        - Technical Skills: 10 = Expert level with project evidence, 5 = Mentioned/Basic, 1 = Barely relevant
        - Soft Skills: Extract from experience descriptions, leadership roles, team projects
        - Experience: Higher score for relevant tech companies, senior roles, long tenure
        - Achievements: Bonus points for hackathons, competitive programming, certifications
        - Resume Quality Scores: Rate the overall quality/strength of each section (0-100)

        Resume Text:
        {res_txt}

        Return only valid JSON without any additional text or formatting:
        """
        
        resp = self.get_completion(prompt)
        try:
            json_start = resp.find('{')
            json_end = resp.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = resp[json_start:json_end]
                return json.loads(json_str)
            else:
                return json.loads(resp)
        except Exception as e:
            raise Exception(f"Error parsing resume JSON: {e}")

    def print_detailed_scores(self, res_data: Dict[str, Any], fname: str):
        print(f"\n{'='*60}")
        print(f"DETAILED SCORING BREAKDOWN - {fname}")
        print(f"{'='*60}")
        
        print(f"Name: {res_data.get('name', 'Unknown')}")
        print(f"Email: {res_data.get('email', 'Not found')}")
        print(f"Phone: {res_data.get('phone', 'Not found')}")
        print(f"Experience: {res_data.get('total_experience', 'N/A')}")
        
        print(f"\nTECHNICAL SKILLS SCORES:")
        print("-" * 40)
        tech_skills = res_data.get('technical_skills', [])
        if tech_skills:
            for skill in tech_skills:
                if isinstance(skill, dict):
                    name = skill.get('skill', 'Unknown')
                    sc = skill.get('score', 0)
                    ev = skill.get('evidence', 'No evidence')
                    print(f"  {name}: {sc}/10 - {ev}")
        else:
            print("  No technical skills found")
        
        print(f"\nSOFT SKILLS SCORES:")
        print("-" * 40)
        soft_skills = res_data.get('soft_skills', [])
        if soft_skills:
            for skill in soft_skills:
                if isinstance(skill, dict):
                    name = skill.get('skill', 'Unknown')
                    sc = skill.get('score', 0)
                    ev = skill.get('evidence', 'No evidence')
                    print(f"  {name}: {sc}/10 - {ev}")
        else:
            print("  No soft skills found")
        
        print(f"\nPROJECT SCORES:")
        print("-" * 40)
        projs = res_data.get('projects', [])
        if projs:
            for proj in projs:
                if isinstance(proj, dict):
                    name = proj.get('name', 'Unknown Project')
                    sc = proj.get('score', 0)
                    techs = proj.get('technologies', [])
                    tech_str = ', '.join(techs) if techs else 'No technologies listed'
                    print(f"  {name}: {sc}/10 - Tech: {tech_str}")
        else:
            print("  No projects found")
        
        print(f"\nWORK EXPERIENCE SCORES:")
        print("-" * 40)
        work_exp = res_data.get('work_experience', [])
        if work_exp:
            for exp in work_exp:
                if isinstance(exp, dict):
                    comp = exp.get('company', 'Unknown')
                    role = exp.get('role', 'Unknown')
                    dur = exp.get('duration', 'Unknown')
                    rel = exp.get('tech_relevance', 0)
                    print(f"  {comp} - {role} ({dur}): {rel}/10 relevance")
        else:
            print("  No work experience found")
        
        print(f"\nACHIEVEMENTS & BONUS POINTS:")
        print("-" * 40)
        achvs = res_data.get('achievements', [])
        if achvs:
            for achv in achvs:
                if isinstance(achv, dict):
                    typ = achv.get('type', 'Unknown')
                    desc = achv.get('description', 'No description')
                    bonus = achv.get('bonus_score', 0)
                    print(f"  {typ}: +{bonus} points - {desc}")
        else:
            print("  No achievements found")
        
        print(f"\nRESUME QUALITY SCORES:")
        print("-" * 40)
        qual = res_data.get('resume_quality_scores', {})
        print(f"  Technical Strength: {qual.get('technical_strength', 0)}/100")
        print(f"  Soft Skills Strength: {qual.get('soft_skills_strength', 0)}/100")
        print(f"  Experience Quality: {qual.get('experience_quality', 0)}/100")
        print(f"  Education Quality: {qual.get('education_quality', 0)}/100")
        print(f"  Bonus Points: {qual.get('bonus_points', 0)}/20")

    def calc_final_score(self, jd: Dict[str, Any], res: Dict[str, Any]) -> Dict[str, Any]:
        qual = res.get('resume_quality_scores', {})
        
        tech_str = qual.get('technical_strength', 0)
        soft_str = qual.get('soft_skills_strength', 0)
        exp_qual = qual.get('experience_quality', 0)
        edu_qual = qual.get('education_quality', 0)
        bonus = qual.get('bonus_points', 0)
        
        sim_data = self.calc_similarity_score(jd, res)
        
        tech_match = sim_data['technical_skills']['match_percentage']
        soft_match = sim_data['soft_skills']['match_percentage']
        edu_match = sim_data['education']['match_percentage']
        
        final = (
            tech_match * 0.3 +
            tech_str * 0.2 +
            exp_qual * 0.2 +
            soft_match * 0.15 +
            soft_str * 0.1 +
            edu_match * 0.05 +
            bonus * 0.5
        )
        
        final = min(final, 100)
        
        recs = []
        if tech_match < 70:
            recs.append("Strengthen technical skills mentioned in JD")
        if exp_qual < 60:
            recs.append("Gain more relevant industry experience")
        if soft_match < 70:
            recs.append("Highlight leadership and communication experiences")
        
        return {
            "final_score": round(final, 2),
            "jd_matching_scores": {
                "technical_skills_match": tech_match,
                "soft_skills_match": soft_match,
                "education_match": edu_match
            },
            "resume_quality_scores": {
                "technical_strength": tech_str,
                "soft_skills_strength": soft_str,
                "experience_quality": exp_qual,
                "education_quality": edu_qual,
                "bonus_points": bonus
            },
            "recommendation": "HIRE" if final >= 75 else "MAYBE" if final >= 60 else "REJECT",
            "strengths": self._get_strengths(res),
            "gaps": self._get_gaps(jd, res),
            "recommendations": recs
        }

    def _get_strengths(self, res: Dict[str, Any]) -> List[str]:
        strengths = []
        
        for skill in res.get('technical_skills', []):
            if isinstance(skill, dict) and skill.get('score', 0) >= 8:
                strengths.append(f"Strong in {skill.get('skill', 'Unknown')}")
        
        achvs = res.get('achievements', [])
        if achvs:
            strengths.append(f"Notable achievements: {len(achvs)} items")
        
        if res.get('total_experience', '0').replace(' years', '').replace('+', '').isdigit():
            exp_yrs = int(res.get('total_experience', '0').replace(' years', '').replace('+', ''))
            if exp_yrs >= 5:
                strengths.append(f"Experienced professional ({exp_yrs}+ years)")
        
        return strengths

    def _get_gaps(self, jd: Dict[str, Any], res: Dict[str, Any]) -> List[str]:
        gaps = []
        
        jd_tech = set([skill.lower() for skill in jd.get('technical_skills', [])])
        
        res_tech = set()
        for skill in res.get('technical_skills', []):
            if isinstance(skill, dict):
                res_tech.add(skill.get('skill', '').lower())
            else:
                res_tech.add(str(skill).lower())
        
        missing = jd_tech - res_tech
        for skill in missing:
            gaps.append(f"Missing: {skill}")
        
        return gaps

    def screen_resume(self, jd_txt: str, res_pdf_path: str) -> Dict[str, Any]:
        jd_data = self.process_jd_cached(jd_txt)
        
        res_txt = self.extract_text_from_pdf(res_pdf_path)
        
        res_data = self.process_resume(res_txt, jd_data)
        
        final_res = self.calc_final_score(jd_data, res_data)
        
        sim_res = self.calc_similarity_score(jd_data, res_data)
        
        return {
            "job_description": jd_data,
            "resume_analysis": res_data,
            "screening_results": final_res,
            "similarity_analysis": sim_res,
            "timestamp": "2025-06-03"
        }

    def batch_screen_resumes(self, jd_file_path: str, res_folder: str) -> List[Dict[str, Any]]:
        jd_txt = self.load_jd_from_file(jd_file_path)
        
        print(f"Job description loaded from: {jd_file_path}")
        
        results = []
        
        for fname in os.listdir(res_folder):
            if fname.endswith('.pdf'):
                print(f"Processing {fname}...")
                try:
                    result = self.screen_resume(jd_txt, os.path.join(res_folder, fname))
                    result['filename'] = fname
                    
                    self.print_detailed_scores(result['resume_analysis'], fname)
                    
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {fname}: {e}")
                    results.append({
                        'filename': fname,
                        'error': str(e),
                        'screening_results': {'final_score': 0, 'recommendation': 'ERROR'}
                    })
        
        results.sort(key=lambda x: x.get('screening_results', {}).get('final_score', 0), reverse=True)
        return results

    def save_to_csv(self, results: List[Dict[str, Any]], fname: str = "output/screening_results.csv"):
        with open(fname, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'filename', 'name', 'email', 'phone', 'final_score', 'recommendation', 'overall_similarity',
                'tech_match_pct', 'soft_match_pct', 'edu_match_pct',
                'tech_strength', 'soft_strength', 'exp_quality', 'edu_quality', 'bonus_points',
                'total_experience', 'strengths', 'gaps', 'recommendations',
                'technical_skills_detailed', 'soft_skills_detailed', 'projects_detailed', 'work_experience_detailed', 'achievements_detailed',
                'error'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = {'filename': result.get('filename', 'Unknown')}
                
                if 'error' in result:
                    row.update({
                        'name': 'Unknown',
                        'email': 'Not found',
                        'phone': 'Not found',
                        'error': result['error'],
                        'final_score': 0,
                        'recommendation': 'ERROR',
                        'technical_skills_detailed': '',
                        'soft_skills_detailed': '',
                        'projects_detailed': '',
                        'work_experience_detailed': '',
                        'achievements_detailed': ''
                    })
                else:
                    screening = result.get('screening_results', {})
                    similarity = result.get('similarity_analysis', {})
                    res_data = result.get('resume_analysis', {})
                    
                    
                    tech_skills_detailed = []
                    for skill in res_data.get('technical_skills', []):
                        if isinstance(skill, dict):
                            tech_skills_detailed.append(f"{skill.get('skill', 'Unknown')}: {skill.get('score', 0)}/10 - {skill.get('evidence', 'No evidence')}")
                    
                    
                    soft_skills_detailed = []
                    for skill in res_data.get('soft_skills', []):
                        if isinstance(skill, dict):
                            soft_skills_detailed.append(f"{skill.get('skill', 'Unknown')}: {skill.get('score', 0)}/10 - {skill.get('evidence', 'No evidence')}")
                    
                    
                    projects_detailed = []
                    for proj in res_data.get('projects', []):
                        if isinstance(proj, dict):
                            techs = ', '.join(proj.get('technologies', []))
                            projects_detailed.append(f"{proj.get('name', 'Unknown')}: {proj.get('score', 0)}/10 - Tech: {techs}")
                    
                    
                    work_exp_detailed = []
                    for exp in res_data.get('work_experience', []):
                        if isinstance(exp, dict):
                            work_exp_detailed.append(f"{exp.get('company', 'Unknown')} - {exp.get('role', 'Unknown')} ({exp.get('duration', 'Unknown')}): {exp.get('tech_relevance', 0)}/10")
                    
                    
                    achievements_detailed = []
                    for achv in res_data.get('achievements', []):
                        if isinstance(achv, dict):
                            achievements_detailed.append(f"{achv.get('type', 'Unknown')}: +{achv.get('bonus_score', 0)} - {achv.get('description', 'No description')}")
                    
                    row.update({
                        'name': res_data.get('name', 'Unknown'),
                        'email': res_data.get('email', 'Not found'),
                        'phone': res_data.get('phone', 'Not found'),
                        'final_score': screening.get('final_score', 0),
                        'recommendation': screening.get('recommendation', 'N/A'),
                        'overall_similarity': similarity.get('overall_similarity', 0),
                        'tech_match_pct': similarity.get('technical_skills', {}).get('match_percentage', 0),
                        'soft_match_pct': similarity.get('soft_skills', {}).get('match_percentage', 0),
                        'edu_match_pct': similarity.get('education', {}).get('match_percentage', 0),
                        'tech_strength': screening.get('resume_quality_scores', {}).get('technical_strength', 0),
                        'soft_strength': screening.get('resume_quality_scores', {}).get('soft_skills_strength', 0),
                        'exp_quality': screening.get('resume_quality_scores', {}).get('experience_quality', 0),
                        'edu_quality': screening.get('resume_quality_scores', {}).get('education_quality', 0),
                        'bonus_points': screening.get('resume_quality_scores', {}).get('bonus_points', 0),
                        'total_experience': res_data.get('total_experience', 'N/A'),
                        'strengths': '; '.join(screening.get('strengths', [])),
                        'gaps': '; '.join(screening.get('gaps', [])),
                        'recommendations': '; '.join(screening.get('recommendations', [])),
                        'technical_skills_detailed': ' | '.join(tech_skills_detailed),
                        'soft_skills_detailed': ' | '.join(soft_skills_detailed),
                        'projects_detailed': ' | '.join(projects_detailed),
                        'work_experience_detailed': ' | '.join(work_exp_detailed),
                        'achievements_detailed': ' | '.join(achievements_detailed),
                        'error': ''
                    })
                
                writer.writerow(row)
        
        print(f"Results saved to {fname}")
