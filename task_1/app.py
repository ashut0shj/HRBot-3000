from flask import Flask, request, render_template_string, redirect, url_for
import os
from werkzeug.utils import secure_filename
from resume_processor import process_resume

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

UPLOAD_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Resume Matcher</title>
</head>
<body>
    <h1>Resume Job Match Scorer</h1>
    <form method="POST" enctype="multipart/form-data">
        <p>Upload your resume (PDF only):</p>
        <input type="file" name="resume" accept=".pdf" required>
        <br><br>
        <input type="submit" value="Upload and Score">
    </form>
</body>
</html>
'''

RESULT_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Resume Match Results</title>
</head>
<body>
    <h1>Resume Match Results</h1>
    
    {% if error %}
        <h2 style="color: red;">Error: {{ error }}</h2>
    {% else %}
        <h2 style="color: {% if match_score >= 70 %}green{% elif match_score >= 50 %}orange{% else %}red{% endif %};">
            Match Score: {{ match_score }}%
        </h2>
        
        <h3>Details:</h3>
        <p><strong>Experience:</strong> {{ resume_data.total_experience }} years</p>
        <p><strong>Location:</strong> {{ resume_data.location }}</p>
        
        {% if matched_skills %}
        <p><strong>Matched Skills:</strong> {{ matched_skills|join(', ') }}</p>
        {% endif %}
        
        {% if resume_data.technical_skills %}
        <p><strong>All Technical Skills:</strong> {{ resume_data.technical_skills|join(', ') }}</p>
        {% endif %}
        
        {% if resume_data.education %}
        <p><strong>Education:</strong> 
        {% for edu in resume_data.education %}
            {{ edu.degree }} from {{ edu.institution }}
        {% endfor %}
        </p>
        {% endif %}
    {% endif %}
    
    <br>
    <a href="/">Upload Another Resume</a>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template_string(UPLOAD_TEMPLATE)
    
    
    if 'resume' not in request.files:
        return render_template_string(RESULT_TEMPLATE, error="No file uploaded")
    
    file = request.files['resume']
    if file.filename == '':
        return render_template_string(RESULT_TEMPLATE, error="No file selected")
    
    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            
            result = process_resume(filepath)
            
            
            os.remove(filepath)
            
            if "error" in result:
                return render_template_string(RESULT_TEMPLATE, error=result["error"])
            else:
                return render_template_string(
                    RESULT_TEMPLATE,
                    match_score=result["match_score"],
                    resume_data=result["resume_data"],
                    matched_skills=result["matched_skills"]
                )
        except Exception as e:
            
            if os.path.exists(filepath):
                os.remove(filepath)
            return render_template_string(RESULT_TEMPLATE, error=f"Processing error: {str(e)}")
    else:
        return render_template_string(RESULT_TEMPLATE, error="Please upload a PDF file only")

if __name__ == '__main__':
    app.run(debug=True)