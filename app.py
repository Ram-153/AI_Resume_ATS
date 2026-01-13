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

# ---------------- NLTK FIX FOR RENDER ----------------
NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

nltk.download("punkt", download_dir=NLTK_DATA_DIR)
nltk.download("punkt_tab", download_dir=NLTK_DATA_DIR)
nltk.download("stopwords", download_dir=NLTK_DATA_DIR)
# ----------------------------------------------------

# Initialize Flask
app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "resumes"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------- Resume Parsing --------
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

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
    error = None

    if request.method == "POST":
        resumes = request.files.getlist("resumes")
        job_desc = request.form.get("job_description", "").strip()

        if not job_desc:
            error = "Please enter job description"
            return render_template("index.html", rankings=[], error=error)

        cleaned_job = preprocess_text(job_desc)

        for resume in resumes:
            if resume.filename == "":
                continue

            file_path = os.path.join(app.config["UPLOAD_FOLDER"], resume.filename)
            resume.save(file_path)

            if resume.filename.lower().endswith(".pdf"):
                text = extract_text_from_pdf(file_path)
            elif resume.filename.lower().endswith(".docx"):
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

        rankings.sort(key=lambda x: x["score"], reverse=True)

    return render_template("index.html", rankings=rankings, error=error)

