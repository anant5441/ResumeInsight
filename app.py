from dotenv import load_dotenv
import base64
import streamlit as st
import os
import io
from PIL import Image 
from PyPDF2 import PdfReader
import pdf2image
import google.generativeai as genai
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

def get_gemini_response(input,pdf_content,prompt):
    model=genai.GenerativeModel('gemini-2.5-pro')
    response=model.generate_content([input,pdf_content[0],prompt])
    return response.text

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        ##Convert PDF to images
        images = pdf2image.convert_from_bytes(uploaded_file.read())

        first_page = images[0]
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()  # encode to base64
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded")  
    
def extract_match_percent(text):
    match = re.search(r'(\d{1,3})\s*%', text)
    return min(100, int(match.group(1))) if match else None
    
# Streamlit App
st.set_page_config(page_title="AI Resume Analyzer 🤖", layout="wide", page_icon="📄")

st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("""
- 📄 Upload Resume  
- 🧠 HR Evaluation  
- 🚀 Skill Suggestions  
- 📊 ATS Match  
- ⚖️ Compare Two Resumes  
- 📥 Download Report  
""")
st.sidebar.info("Developed by Anant Bhardwaj using Gemini Pro ✨")

st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>AI Resume Analyzer 🤖</h1>
    <p style='text-align: center;'>Powered by Gemini | ATS + HR Evaluation</p>
    <hr style='border: 1px solid #4CAF50;'/>
""", unsafe_allow_html=True)
st.header("ATS Tracking System")
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("📄 Upload your Resume (PDF)", type=["pdf"])

with col2:
    input_text = st.text_area("🧾 Paste Job Description", height=220)


if uploaded_file is not None:
    st.write("PDF Uploaded Successfully")

submit1 = st.button("🧠 Tell Me About the Resume")
submit2 = st.button("🚀 How Can I Improve My Skills")
submit3 = st.button("📊 Percentage Match")

input_prompt1 = """
You are an experienced Technical Human Resource Manager and hiring expert who specializes in analyzing resumes for technology and data-driven roles. 
Your task is to carefully evaluate the candidate's resume in comparison with the provided job description.

Provide your response in a professional, well-structured, and HR-style review format using the following sections:

1. **Overall Impression:**
   - Provide a concise summary of the candidate’s overall profile, tone, and first impression.
   - Comment on whether the resume appears tailored for the role or generic.

2. **Core Strengths:**
   - Highlight the key strengths and unique selling points of the candidate (e.g., technical expertise, domain knowledge, impactful projects, internships, or achievements).
   - Focus on how these strengths support the requirements in the job description.

3. **Technical and Professional Alignment:**
   - Evaluate how well the candidate’s technical skills, tools, and experience align with the key requirements in the job description.
   - Mention specific overlaps between resume keywords and job requirements (e.g., both mention “Python,” “Data Analysis,” or “Machine Learning”).

4. **Gaps or Weak Areas:**
   - Identify any noticeable skill gaps, limited experience, or missing details that could weaken the application.
   - Comment on areas that could benefit from clearer articulation or stronger evidence (e.g., quantifiable results, leadership roles, certifications).

5. **Communication & Formatting Quality:**
   - Assess the overall clarity, formatting, structure, and readability of the resume.
   - Mention if it’s well-organized, ATS-friendly, and free of unnecessary jargon.

6. **Overall Evaluation:**
   - Conclude with a short summary of your hiring assessment (e.g., “Strong Fit,” “Moderate Fit,” or “Needs Improvement”).
   - Give a final recommendation on whether the resume would likely pass the HR screening and proceed to the technical interview stage.

Keep the tone constructive, detailed, and professional — as if you were writing a real HR evaluation report for a candidate.
"""


input_prompt2 = """
You are a career development expert and technical mentor specializing in data science, software engineering, and technology hiring.
Your task is to carefully analyze the candidate's resume in comparison with the provided job description.

Your goal is to give the candidate a professional, step-by-step improvement plan to make their resume stronger and better aligned with the target role.

Please follow this structured output format:

1. **Overall Summary:**
   - Provide a short summary of how well the current resume aligns with the job description.
   - Mention key areas where the candidate stands out.

2. **Skill Gap Analysis:**
   - List the technical, analytical, or domain-specific skills that are missing or underdeveloped compared to the job description.
   - Include both hard skills (e.g., Python, TensorFlow, SQL, Power BI) and soft skills (e.g., teamwork, leadership, problem-solving).

3. **Improvement Recommendations:**
   - For each missing or weak skill, provide clear and practical advice on how to build or improve it.
   - Recommend relevant certifications, online courses (from Coursera, edX, Udemy, etc.), open-source contributions, or personal projects that would strengthen the resume.

4. **Resume Enhancement Tips:**
   - Suggest how the candidate can better present their achievements, quantify impact (using metrics or results), and structure sections more effectively for ATS readability.
   - Mention how to optimize keywords and formatting to pass ATS filters.

5. **Career Growth Roadmap:**
   - Create a short roadmap divided into:
       - **Short-Term (1–3 months):** Immediate steps to improve or learn specific tools/skills.
       - **Long-Term (4–12 months):** Broader professional development goals like internships, publications, advanced certifications, or leadership experience.

Conclude with a motivational message that encourages continuous learning and alignment with industry trends.
"""


input_prompt3 = """
You are a highly accurate and intelligent Applicant Tracking System (ATS) used by HR teams to evaluate resumes for technical positions.
Your task is to compare the candidate’s resume with the provided job description and generate a structured compatibility analysis.

You must act as a real ATS scanner that understands keyword relevance, job role fit, and experience mapping.

Please provide the analysis in the following detailed and structured format:

1. **ATS Match Score:**
   - Provide a numerical percentage (0–100%) indicating how well the resume matches the job description.
   - Explain briefly what factors contributed to this score (e.g., strong alignment in skills, weak alignment in experience).

2. **Keyword & Skill Match Analysis:**
   - List important keywords and skills *present* in the resume that match the job description.
   - Then, list *missing or weakly represented* keywords/skills that are required for the role but absent or underemphasized in the resume.
   - Focus on both technical (e.g., Python, machine learning, cloud platforms) and non-technical keywords (e.g., collaboration, communication).

3. **Experience and Role Fit:**
   - Evaluate how well the candidate’s professional and academic experience aligns with the job requirements.
   - Comment on the relevance of previous projects, internships, or positions.

4. **Education and Certifications Match:**
   - Check if the educational background or certifications meet the job’s qualifications.

5. **Final Evaluation:**
   - Summarize the overall job fit.
   - Mention whether the candidate would likely pass the ATS stage (e.g., “High chance of shortlisting”, “Moderate chance”, or “Needs optimization”).
   - Offer 2–3 quick tips to improve the ATS score (e.g., adding missing keywords, reordering experience, optimizing skill phrasing).

Format the output cleanly using headings, bullet points, and emphasis for readability.
"""

comparison_prompt = """
You are an experienced Technical HR Manager and career advisor. 
You are asked to **compare two resumes** for a similar technical/data-driven role based on a provided job description. 
The first resume belongs to Candidate A and the second belongs to Candidate B. 

Please provide a detailed comparison covering the following sections:

1. **Overall Impression:**
   - Briefly summarize the overall profile and tone of each candidate.
   - Comment on which resume appears better tailored for the role.

2. **Core Strengths:**
   - Highlight the key strengths and unique selling points of both candidates.
   - Mention technical expertise, projects, internships, achievements, and domain knowledge.

3. **Technical and Professional Alignment:**
   - Compare how well each candidate’s skills, tools, and experience align with the job description.
   - Mention specific overlaps with required skills and qualifications.

4. **Gaps or Weak Areas:**
   - Identify skill gaps, limited experience, or missing details for each candidate.
   - Suggest which areas could weaken their application.

5. **Communication & Formatting Quality:**
   - Assess clarity, structure, readability, and ATS-friendliness for both resumes.

6. **Recommendation:**
   - Conclude by stating which candidate is a better fit for the role.
   - Provide a short explanation for the recommendation.

Keep the tone **professional, constructive, and detailed**, as if writing a real HR comparative report.
"""


pdf_content = input_pdf_setup(uploaded_file) if uploaded_file else None

if submit1:
    if uploaded_file:
        with st.spinner("Analyzing resume from HR perspective..."):
            hr_feedback = get_gemini_response(input_text, pdf_content, input_prompt1)
        tab1, tab2, tab3 = st.tabs(["🧠 HR Feedback", "🚀 Skill Gaps", "📊 ATS Match"])
        with tab1:
            st.subheader("HR Analysis Report")
            st.markdown(hr_feedback)
    else:
        st.error("Please upload a PDF file.")

elif submit2:
    if uploaded_file:
        with st.spinner("Analyzing skill improvement areas..."):
            skill_suggestions = get_gemini_response(input_text, pdf_content, input_prompt2)
        tab1, tab2, tab3 = st.tabs(["🧠 HR Feedback", "🚀 Skill Gaps", "📊 ATS Match"])
        with tab2:
            st.subheader("Skill Enhancement Suggestions")
            st.markdown(skill_suggestions)

        # Extract top skills
        skill_extract_prompt = "List top 10 hard and soft skills found in this resume as bullet points."
        skills_text = get_gemini_response("", pdf_content, skill_extract_prompt)
        st.markdown("### 🧩 Key Skills Identified:")
        skills_list = [s.strip("- ").strip() for s in skills_text.split("\n") if s.strip()]
        st.write(", ".join([f"🟢 {skill}" for skill in skills_list]))

elif submit3:
    if uploaded_file:
        with st.spinner("Calculating ATS Match Score..."):
            match_report = get_gemini_response(input_text, pdf_content, input_prompt3)
        tab1, tab2, tab3 = st.tabs(["🧠 HR Feedback", "🚀 Skill Gaps", "📊 ATS Match"])
        with tab3:
            st.subheader("ATS Evaluation Report")
            st.markdown(match_report)

            # Show progress bar
            match_percent = extract_match_percent(match_report)
            if match_percent:
                st.progress(match_percent / 100)
                st.write(f"**Match Score: {match_percent}%**")
            else:
                st.warning("Couldn't detect a numeric percentage in the response.")

            # Missing keywords wordcloud
            missing_keywords = "Python SQL Machine Learning Leadership" # example placeholder
            st.markdown("### 📌 Missing Keywords Cloud")
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(missing_keywords)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)

            # Radar chart for skills
            st.markdown("### 📊 Skills Radar Chart")
            skill_data = pd.DataFrame({
                "Skill": ["Python","SQL","ML","Communication","Leadership"],
                "Match": [90, 70, 80, 60, 50]
            })
            fig = px.line_polar(skill_data, r='Match', theta='Skill', line_close=True)
            st.plotly_chart(fig)

# st.markdown("### ⚖️ Compare Two Resumes")
# col1, col2 = st.columns(2)
# with col1:
#     resume1 = st.file_uploader("Upload Resume 1 (PDF)", type=["pdf"], key="res1")
# with col2:
#     resume2 = st.file_uploader("Upload Resume 2 (PDF)", type=["pdf"], key="res2")

# if st.button("Compare Resumes"):
#         if resume1 and resume2:
#             with st.spinner("Analyzing both resumes..."):
#                 pdf1_content = input_pdf_setup(resume1)
#                 pdf2_content = input_pdf_setup(resume2)
#                 report1 = get_gemini_response(input_text, pdf1_content, input_prompt3)
#                 report2 = get_gemini_response(input_text, pdf2_content, input_prompt3)
#                 score1 = extract_match_percent(report1) or 0
#                 score2 = extract_match_percent(report2) or 0
#                 hr1 = get_gemini_response(input_text, pdf1_content, input_prompt1)
#                 hr2 = get_gemini_response(input_text, pdf2_content, input_prompt1)
#             # Comparison table
#             comp_df = pd.DataFrame({
#                 "Resume": ["Resume 1", "Resume 2"],
#                 "ATS Score (%)": [score1, score2],
#                 "Overall Recommendation": [
#                     "Better Fit" if score1 >= score2 else "Moderate Fit",
#                     "Better Fit" if score2 > score1 else "Moderate Fit"
#                 ]
#             })
#             st.table(comp_df)
#             # HR feedback tabs
#             tab1, tab2 = st.tabs(["Resume 1 Details", "Resume 2 Details"])
#             with tab1:
#                 st.subheader("Resume 1 HR Feedback")
#                 st.markdown(hr1)
#             with tab2:
#                 st.subheader("Resume 2 HR Feedback")
#                 st.markdown(hr2)

#             pdf1_content = get_pdf_text(resume1)
#             pdf2_content = get_pdf_text(resume2)
#             with st.spinner("Comparing both resumes..."):
#                 comparison_result = get_gemini_response(
#                     input_text, 
#                     [pdf1_content, pdf2_content],  # send both resumes
#                     comparison_prompt
#                 )

#             st.subheader("📄 Resume Comparison Report")
#             st.markdown(comparison_result)
#             better_resume = "Resume 1" if score1 >= score2 else "Resume 2"
#             st.success(f"✅ Based on ATS score and HR analysis, **{better_resume}** is a stronger resume for this job.")
#         else:
#             st.error("Please upload both resumes for comparison.")

st.title("📊 Resume Comparison")

st.write("Upload **two resumes** to compare and get detailed analysis including similarities, strengths, and improvement points.")

pdf1 = st.file_uploader("Upload First Resume", type=["pdf"], key="compare_pdf1")
pdf2 = st.file_uploader("Upload Second Resume", type=["pdf"], key="compare_pdf2")

input_text = st.text_area("Enter the Job Description (JD)")

if st.button("Compare Resumes"):
    if pdf1 and pdf2:
        with st.spinner("🔍 Analyzing and comparing resumes..."):
            pdf1_content = get_pdf_text(pdf1)
            pdf2_content = get_pdf_text(pdf2)

            comparison_prompt = f"""
            You are an expert HR evaluator.
            Compare the two resumes below based on this job description: "{input_text}".

            - Candidate 1's Resume:
            {pdf1_content}
            - Candidate 2's Resume:
            {pdf2_content}

            Provide a structured comparison:
            1. Summary of both candidates.
            2. Strengths and weaknesses of each.
            3. Who is a better fit for the JD and why.
            4. Any missing skills or improvements required.
            """

            # ✅ Fix: Pass merged text instead of list to Gemini
            comparison_result = get_gemini_response(input_text, pdf1_content + "\n\n" + pdf2_content, comparison_prompt)

            st.subheader("📋 Comparison Result")
            st.write(comparison_result)
    else:
        st.warning("⚠️ Please upload both resumes for comparison.")

st.markdown("### 💬 Ask AI about your resume")
user_question = st.text_input("Enter your question here:")
if st.button("Ask AI"):
    if uploaded_file and user_question.strip():
        with st.spinner("AI is analyzing your resume and preparing an answer..."):
            answer = get_gemini_response(user_question, pdf_content, "")
        st.success("✅ Response Ready")
        st.write(answer)
    elif not uploaded_file:
        st.error("Please upload a PDF file first.")
    else:
        st.warning("Please enter a question to ask.")


if uploaded_file and (submit1 or submit2 or submit3):
    st.download_button(
        "📥 Download Full Report",
        data=f"Job Description:\n{input_text}\n\nAnalysis:\n{hr_feedback if submit1 else skill_suggestions if submit2 else match_report}",
        file_name="AI_Resume_Analysis_Report.txt",
        mime="text/plain"
    )   

st.markdown("""
<hr/>
<p style='text-align:center;'>💼 Built with ❤️ by <b>Anant Bhardwaj</b></p>
""", unsafe_allow_html=True)