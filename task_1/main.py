from ResumeScreeningTool import ResumeScreeningTool
import os
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
model = 'gemini-1.5-flash'

def main():
    tool = ResumeScreeningTool(api_key=api_key, model=model)
    
    jd_file = "job_description.txt"
    
    
    try:
        print("Starting batch processing...")
        results = tool.batch_screen_resumes(jd_file, "resumes/")
        
        print("\nBatch Screening Results:")
        print("=" * 50)
        
        for i, result in enumerate(results[:10], 1):
            if 'error' not in result:
                sc = result['screening_results']['final_score']
                rec = result['screening_results']['recommendation']
                sim = result.get('similarity_analysis', {}).get('overall_similarity', 0)
                print(f"{i}. {result['filename']}: {sc}% (AI) | {sim}% (Code) | ({rec})")
            else:
                print(f"{i}. {result['filename']}: ERROR - {result['error']}")
        
        tool.save_to_csv(results)
        print("\nProcessing completed successfully!")
        
    except Exception as e:
        print(f"Batch processing error: {e}")


if __name__ == "__main__":
    main()