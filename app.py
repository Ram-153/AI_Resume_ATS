from flask import Flask, render_template, request
import os
import PyPDF2
import docx
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

UPLOAD_FOLDER = "resumes"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------- Resume Parsing --------

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# -------- NLP Preprocessing --------

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_words)

# -------- ATS Scoring --------

def calculate_ats_score(resume_text, job_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_text])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(similarity * 100, 2)

# -------- Routes --------

@app.route("/", methods=["GET", "POST"])
def index():
    rankings = []

    if request.method == "POST":
        resumes = request.files.getlist("resumes")
        job_desc = request.form["job_description"]

        cleaned_job = preprocess_text(job_desc)

        for resume in resumes:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], resume.filename)
            resume.save(file_path)

            if resume.filename.endswith(".pdf"):
                text = extract_text_from_pdf(file_path)
            elif resume.filename.endswith(".docx"):
                text = extract_text_from_docx(file_path)
            else:
                continue

            cleaned_resume = preprocess_text(text)
            score = calculate_ats_score(cleaned_resume, cleaned_job)

            rankings.append({
                "name": resume.filename,
                "score": score,
                "status": "Selected" if score >= 60 else "Not Selected"
            })

        rankings = sorted(rankings, key=lambda x: x["score"], reverse=True)

    return render_template("index.html", rankings=rankings)

if __name__ == "__main__":
    app.run(debug=True)
