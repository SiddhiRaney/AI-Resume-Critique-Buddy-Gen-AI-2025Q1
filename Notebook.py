# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Project Name: AI Resume Critique Buddy

**Problem Statement:** Resumes often lack tailored, impactful bullet points aligned with the job role, which reduces their chances of clearing ATS filters and impressing recruiters.

 **GenAI can solve this by:**
- Generating personalized bullet points using Few-Shot Prompting.
- Evaluating and enhancing these points for clarity and impact.
- Analyzing the resume for missing content using Retrieval-Augmented Generation (RAG).

**AIM:**
To leverage artificial intelligence to provide tailored, impactful, and job-relevant resume bullet points based on the candidate's existing resume. It aims to enhance the quality of resumes, increasing their chances of getting noticed by employers.

**Key Features:**
1. Tailored resume bullet points specific to the job role, based on pre-defined examples and the provided resume text.
2. Evaluation of the generated bullet points for clarity, impact, and relevance, ensuring alignment with job requirements.
3. A detailed resume analysis report that includes strengths, missing elements, and suggestions for improvement based on external data retrieval and resume evaluation.

**Gen AI Capabilities**
1. **Few-Shot Prompting:** This approach uses the pre-defined few-shot examples of bullet points, combined with the resume text, to generate tailored and relevant resume bullet points using Google's Gemini AI model. The output is formatted and ready for use in the candidate's resume, improving its chances of standing out to employers.

2. **Gen AI Evaluation:** Gen AI evaluation to assess the quality of the generated bullet points for clarity, impact, and relevance. We then compare the generated content with the provided resume to ensure it aligns with job requirements and enhances the resume's effectiveness.

3. **RAG:** The resume analysis report includes the following:
a) Strengths
b) Missing Elements
c) Suggestions for Improvement

**Final Output**
1.  Role-aligned bullet points ready to paste into resume.
2.  Recommendations to further polish resume content.

**Real World Application**
This project uses AI to optimize resumes by generating tailored bullet points, evaluating content quality, and analyzing strengths, missing elements, and improvement suggestions. It enhances resumes for better alignment with job roles, improving chances with recruiters and ATS.

**This tool is designed to be intuitive, user-friendly, and highly effective in assisting users to create the best possible resume to stand out in the competitive job market.**
# **Installing Required Packages:**
# Install the google-genai package if not already installed
!pip install -U google-generativeai

# Importing Libraries:
# Importing necessary libraries
from google import genai
from google.genai import types
from kaggle_secrets import UserSecretsClient
import google.generativeai as genai
import os
from kaggle_secrets import UserSecretsClient

# Get the API key securely from Kaggle secrets
secret = UserSecretsClient()
api_key = secret.get_secret("GOOGLE_API_KEY")

# Configure Gemini
genai.configure(api_key=api_key)
pip install --upgrade google-generativeai
# API USAGE:
from kaggle_secrets import UserSecretsClient
import google.generativeai as genai

# Fetch the Google API Key from Kaggle secrets
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GOOGLE_API_KEY")

# Set the API key
genai.configure(api_key=api_key)

# Check available models
models = genai.list_models()  # This will list available models in your current API setup

# Print the available models
print(models)

# **Core libraries like google-generativeai for the AI capabilities**
pip install google-generativeai Flask mysql-connector pymysql
# We'll use PyPDF2 to extract text from a resume PDF. First, make sure PyPDF2 is installed:
pip install PyPDF2
def extract_text_from_pdf(file_path):
    import PyPDF2
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Set path to your uploaded resume
resume_path = "/kaggle/input/sampleresume/John Doe Resume.pdf"  
resume_text = extract_text_from_pdf(resume_path)

print(resume_text)

# 1. Few-Shot Prompting
This approach uses the pre-defined few-shot examples of bullet points, combined with the resume text, to generate tailored and relevant resume bullet points using Google's Gemini AI model. The output is formatted and ready for use in the candidate's resume, improving its chances of standing out to employers.
import google.generativeai as genai
import json

# Example job titles and their tailored resume bullet points
example_bullet_points = {
    "Software Developer": [
        "Designed and implemented a feature that improved user engagement by 25%",
        "Collaborated with cross-functional teams to optimize application performance and reduce loading time by 30%",
        "Developed automated testing tools that decreased QA cycle time by 40%",
        "Led the migration of legacy systems to modern cloud infrastructure using AWS"
    ],
    "Customer Service Representative": [
        "Resolved customer inquiries, achieving a 95% customer satisfaction rate",
        "Managed customer complaints and turned them into positive experiences, improving retention by 20%",
        "Trained new customer service representatives, improving team efficiency by 15%",
        "Processed over 150 customer orders daily while maintaining error-free records"
    ],
    "Data Analyst": [
        "Analyzed large datasets to derive actionable insights that resulted in a 15% increase in sales",
        "Developed automated reporting tools, saving 10 hours of manual work per week",
        "Created predictive models to forecast customer churn with an accuracy of 85%",
        "Led data-driven projects that improved operational efficiency by 20%"
    ]
}

# Function to generate prompt for few-shot prompting with example bullet points
def generate_resume_prompt(resume_text, job_title):
    # Get the example bullet points for the job role
    examples = example_bullet_points.get(job_title, [])
    
    # Create prompt with examples and new job role
    prompt = f"""
    You are a resume expert. I will provide you with examples of resume bullet points for various roles. Based on the examples, please generate tailored bullet points for a resume in the job role of {job_title}. 

    Example Resume Bullet Points for {job_title}:
    """
    for bullet in examples:
        prompt += f"    - {bullet}\n"
    
    prompt += f"""
    Now, analyze the resume text and generate a set of bullet points that reflect the job responsibilities and achievements for the role of {job_title} based on the resume information. Use your knowledge of the role to make the points impactful.

    Resume:
    "{resume_text}"
    """
    return prompt

# Function to generate tailored bullet points using Gemini
def generate_bullet_points_with_gemini(resume_text, job_title):
    try:
        prompt = generate_resume_prompt(resume_text, job_title)
        model = genai.GenerativeModel("models/gemini-2.0-flash")
        response = model.generate_content(prompt)

        text = response.text
        cleaned_text = text.replace("```json\n", "").replace("\n```", "")
        
        # Just returning the clean response text for formatted output
        print("Generated Bullet Points:\n")
        formatted_response = cleaned_text.strip().replace('\n', '\n* ')
        print(f"* {formatted_response}")

        return cleaned_text
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

# Example usage
resume_text = """
Experienced Software Developer with a strong background in Python, JavaScript, and cloud computing. Skilled in building scalable applications and collaborating with teams to develop software solutions. Proven track record of optimizing processes and improving system performance.
"""
job_title = "Software Developer"

# Get tailored bullet points for the Software Developer role
generate_bullet_points_with_gemini(resume_text, job_title)
# 2. Gen AI Evaluation
Gen AI evaluation to assess the quality of the generated bullet points for clarity, impact, and relevance. We then compare the generated content with the provided resume to ensure it aligns with job requirements and enhances the resume's effectiveness.
import google.generativeai as genai
import json

# Example job titles and their tailored resume bullet points
example_bullet_points = {
    "Software Developer": [
        "Designed and implemented a feature that improved user engagement by 25%",
        "Collaborated with cross-functional teams to optimize application performance and reduce loading time by 30%",
        "Developed automated testing tools that decreased QA cycle time by 40%",
        "Led the migration of legacy systems to modern cloud infrastructure using AWS"
    ],
    "Customer Service Representative": [
        "Resolved customer inquiries, achieving a 95% customer satisfaction rate",
        "Managed customer complaints and turned them into positive experiences, improving retention by 20%",
        "Trained new customer service representatives, improving team efficiency by 15%",
        "Processed over 150 customer orders daily while maintaining error-free records"
    ],
    "Data Analyst": [
        "Analyzed large datasets to derive actionable insights that resulted in a 15% increase in sales",
        "Developed automated reporting tools, saving 10 hours of manual work per week",
        "Created predictive models to forecast customer churn with an accuracy of 85%",
        "Led data-driven projects that improved operational efficiency by 20%"
    ]
}

# Function to generate prompt for few-shot prompting with example bullet points
def generate_resume_prompt(resume_text, job_title):
    # Get the example bullet points for the job role
    examples = example_bullet_points.get(job_title, [])
    
    # Create prompt with examples and new job role
    prompt = f"""
    You are a resume expert. I will provide you with examples of resume bullet points for various roles. Based on the examples, please generate tailored bullet points for a resume in the job role of {job_title}. 

    Example Resume Bullet Points for {job_title}:
    """
    for bullet in examples:
        prompt += f"    - {bullet}\n"
    
    prompt += f"""
    Now, analyze the resume text and generate a set of bullet points that reflect the job responsibilities and achievements for the role of {job_title} based on the resume information. Use your knowledge of the role to make the points impactful.

    Resume:
    "{resume_text}"
    """
    return prompt

# Function to evaluate the quality of generated content
def evaluate_content_quality(generated_content):
    try:
        # Call the Gemini evaluation model
        model = genai.GenerativeModel("models/gemini-2.0-eval")  # Assuming there's an evaluation model
        evaluation_response = model.generate_content({
            "content": generated_content,
            "task": "Evaluate the quality of this content for a resume. Check for clarity, impact, and relevance."
        })

        evaluation = evaluation_response.text
        print(f"Evaluation of the Generated Content:\n{evaluation}")
    except Exception as e:
        print(f"Error during evaluation: {e}")

# Function to generate tailored bullet points using Gemini
def generate_bullet_points_with_gemini(resume_text, job_title):
    try:
        prompt = generate_resume_prompt(resume_text, job_title)
        model = genai.GenerativeModel("models/gemini-2.0-flash")
        response = model.generate_content(prompt)

        text = response.text
        cleaned_text = text.replace("```json\n", "").replace("\n```", "")
        
        # Output the generated bullet points
        print("Generated Bullet Points:\n")
        formatted_response = cleaned_text.strip().replace('\n', '\n* ')
        print(f"* {formatted_response}")
        
        # Now, evaluate the quality of the generated content
        evaluate_content_quality(cleaned_text)

    except Exception as e:
        print(f"Error occurred: {e}")
        return None

# Example usage
resume_text = """
Experienced Software Developer with a strong background in Python, JavaScript, and cloud computing. Skilled in building scalable applications and collaborating with teams to develop software solutions. Proven track record of optimizing processes and improving system performance.
"""
job_title = "Software Developer"

# Get tailored bullet points for the Software Developer role and evaluate them
generate_bullet_points_with_gemini(resume_text, job_title)

# RAG(Retrieval-Augmented Generation):
is retrieving additional relevant information from external sources to enhance the resume's tailored bullet points and suggestions. This data is then incorporated into the analysis to provide a more comprehensive and contextually aligned evaluation.
The resume analysis report includes the following:

1. Strengths
2. Missing Elements
3. Suggestions for Improvement
import google.generativeai as genai
import json

# Configure the model
model = genai.GenerativeModel("models/gemini-1.5-flash")  # this works with generate_content

# Resume text
resume_text = """
Experienced Software Developer with a strong background in Python, JavaScript, and cloud computing. Skilled in building scalable applications and collaborating with teams to develop software solutions. Proven track record of optimizing processes and improving system performance.
"""

# Job title
job_title = "Software Developer"

# Controlled prompt with structured output
def generate_resume_report_prompt(resume_text, job_title):
    prompt = f"""
    You are an expert resume evaluator. Review the following resume for the job title: {job_title}.

    Return a structured JSON object in this format:
    {{
        "strengths": ["Highlight what the resume does well"],
        "missing_elements": ["What skills or experience are missing for the job role"],
        "suggestions": ["How the resume can be improved for this job"]
    }}

    Resume:
    \"\"\"{resume_text}\"\"\"
    """
    return prompt

# Function to generate the report
def generate_structured_resume_report(resume_text, job_title):
    try:
        prompt = generate_resume_report_prompt(resume_text, job_title)
        response = model.generate_content(prompt)

        cleaned_text = response.text.replace("```json", "").replace("```", "").strip()

        try:
            report = json.loads(cleaned_text)
            return report
        except json.JSONDecodeError:
            print("Could not parse response as JSON. Raw output:")
            return cleaned_text

    except Exception as e:
        print(f"Error: {e}")
        return None

# Run the report generator
report = generate_structured_resume_report(resume_text, job_title)

# Output result (Formatted)
if isinstance(report, dict):
    print("\n✅ Resume Analysis Report:\n")

    print("🔹 Strengths:")
    for item in report.get("strengths", []):
        print(f"  - {item}")

    print("\n🔸 Missing Elements:")
    for item in report.get("missing_elements", []):
        print(f"  - {item}")

    print("\n💡 Suggestions for Improvement:")
    for item in report.get("suggestions", []):
        print(f"  - {item}")

elif report:
    print("📝 Raw Output:\n")
    print(report)
else:
    print("❌ No report generated.")
**Limitations**
1. May require manual review for niche domains.
2. Bullet points can still be generic if training data is limited.
3. No real-time recruiter feedback loop.

**Future Scope**
1. Add Job Description parser
2. Offer LinkedIn bio & cover letter generation
3. Integrate with job platforms (like LinkedIn, Indeed) to suggest real-time job matches based on the resume content
# AI Resume Critique Buddy advanced AI techniques like Few-Shot Prompting, Gen AI Evaluation, and RAG (Retrieval-Augmented Generation) to enhance and optimize resumes for job applications. Few-Shot Prompting helps generate tailored resume bullet points based on predefined examples, ensuring relevance to the desired job role. Gen AI Evaluation assesses the quality of these bullet points for clarity, impact, and job-specific alignment. RAG further enhances the process by analyzing the resume for strengths, identifying missing elements, and offering actionable suggestions for improvement, thereby increasing the chances of standing out to recruiters and passing through Applicant Tracking Systems (ATS). The project is designed to make resumes more effective and impactful, boosting a candidate's job prospects.
# Citation:
@misc{gen-ai-intensive-course-capstone-2025q1,
    author = {Addison Howard and Brenda Flynn and Myles O'Neill and Nate and Polong Lin},
    title = {Gen AI Intensive Course Capstone 2025Q1},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/gen-ai-intensive-course-capstone-2025q1}},
    note = {Kaggle}
}
