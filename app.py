import os
import re
import json
from datetime import datetime
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Config ----------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}
HISTORY_FILE = os.path.join("data", "history.csv")  # history stored here
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("data", exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024  # 30 MB

# ---------- Job keywords (editable) ----------
JOB_KEYWORDS = {
    "Data Scientist": ["python", "machine learning", "pandas", "numpy", "scikit-learn", "statistics", "eda", "feature engineering"],
    "Data Analyst": ["sql", "excel", "powerbi", "tableau", "data analysis", "reporting", "pivot"],
    "Machine Learning Engineer": ["python", "tensorflow", "pytorch", "model deployment", "docker", "mlops", "api"],
    "Software Developer": ["java", "python", "c++", "git", "rest api", "oop", "algorithms"],
    "Android Developer": ["kotlin", "android studio", "java", "xml", "jetpack", "firebase"],
    "Web Developer": ["html", "css", "javascript", "react", "nodejs", "rest api"],
    "DevOps Engineer": ["docker", "kubernetes", "ci/cd", "ansible", "terraform", "aws"],
    "Cloud Engineer": ["aws", "azure", "gcp", "cloud", "ec2", "s3", "lambda"],
    "NLP Engineer": ["nlp", "transformers", "bert", "nlp pipeline", "spacy"],
    "Computer Vision Engineer": ["opencv", "cnn", "tensorflow", "pytorch", "object detection"],
    "Business Analyst": ["sql", "excel", "stakeholder", "dashboard", "powerbi"],
    "QA Engineer": ["testing", "selenium", "automation", "pytest", "test cases"],
    "Database Administrator": ["mysql", "postgresql", "database", "backup", "replication"],
    "Cloud DevOps": ["aws", "docker", "kubernetes", "jenkins", "ci/cd"],
    "Backend Developer": ["django", "flask", "rest api", "database", "orm"]
}

# ---------- Helpers ----------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(path):
    text = ""
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                try:
                    p = page.extract_text()
                    if p:
                        text += p + "\n"
                except Exception:
                    continue
    except Exception:
        return ""
    return text

def clean_text(t):
    if not t:
        return ""
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def calculate_ats_score(resume_text, job_text):
    resume_text = resume_text or ""
    job_text = job_text or ""
    try:
        vec = TfidfVectorizer(stop_words="english")
        mat = vec.fit_transform([resume_text, job_text])
        score = cosine_similarity(mat[0:1], mat[1:2])[0][0]
    except Exception:
        score = 0.0
    return round(float(score * 100), 2)

def extract_top_keywords(text, top_n=20):
    text = text or ""
    try:
        vec = TfidfVectorizer(stop_words="english", max_features=2000)
        tfidf = vec.fit_transform([text])
        feats = vec.get_feature_names_out()
        scores = tfidf.toarray()[0]
        idx = scores.argsort()[::-1][:top_n]
        keywords = [feats[i] for i in idx if scores[i] > 0]
        return keywords
    except Exception:
        words = re.findall(r"\b[a-zA-Z0-9\+\-#]+\b", text.lower())
        freq = pd.Series(words).value_counts()
        return freq.head(top_n).index.tolist()

def match_keywords(keywords, resume_text):
    res = (resume_text or "").lower()
    matched = []
    missing = []
    for kw in keywords:
        kw_clean = kw.lower()
        pattern = r"\b" + re.escape(kw_clean) + r"\b"
        if re.search(pattern, res):
            matched.append(kw)
        else:
            if kw_clean in res:
                matched.append(kw)
            else:
                missing.append(kw)
    return matched, missing

def role_suitability_score(role_name, resume_text):
    keywords = JOB_KEYWORDS.get(role_name, [])
    if not keywords:
        return 0.0, [], []
    matched, missing = match_keywords(keywords, resume_text)
    score = round(len(matched) / len(keywords) * 100.0, 2)
    return score, matched, missing

def suggest_for_missing(missing):
    mapping = {
        "python":"https://www.coursera.org/courses?query=python",
        "machine learning":"https://www.coursera.org/learn/machine-learning",
        "docker":"https://www.coursera.org/courses?query=docker",
        "aws":"https://www.coursera.org/courses?query=aws",
        "sql":"https://www.coursera.org/courses?query=sql",
        "tensorflow":"https://www.coursera.org/courses?query=tensorflow",
        "pandas":"https://pandas.pydata.org/docs/getting_started/index.html",
        "react":"https://reactjs.org/tutorial/tutorial.html"
    }
    out = {}
    for kw in missing:
        lk = kw.lower()
        for k, u in mapping.items():
            if k in lk or lk in k:
                out[kw] = u
                break
    return out

def append_history(role_label, score):
    # Create history CSV if not exists, then append
    row = {"timestamp": datetime.now().isoformat(), "role": role_label, "score": float(score)}
    try:
        if not os.path.exists(HISTORY_FILE):
            dfh = pd.DataFrame([row])
            dfh.to_csv(HISTORY_FILE, index=False)
        else:
            dfh = pd.DataFrame([row])
            dfh.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
    except Exception:
        pass

def get_trend_for_role(role_label, max_points=30):
    if not os.path.exists(HISTORY_FILE):
        return [], []
    try:
        df = pd.read_csv(HISTORY_FILE)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df[df['role'].astype(str) == str(role_label)]
        df = df.sort_values('timestamp').tail(max_points)
        labels = df['timestamp'].dt.strftime("%Y-%m-%d %H:%M").tolist()
        scores = df['score'].astype(float).tolist()
        return labels, scores
    except Exception:
        return [], []

# ---------- Routes ----------
@app.route("/", methods=["GET"])
def index():
    roles = list(JOB_KEYWORDS.keys())
    jobs_list = []
    jobs_path = os.path.join("data", "jobs.csv")
    if os.path.exists(jobs_path):
        try:
            dfj = pd.read_csv(jobs_path)
            for _, r in dfj.iterrows():
                jobs_list.append({
                    "title": str(r.get("title","")),
                    "description": str(r.get("description","")) if "description" in r else str(r.get("skills",""))
                })
        except Exception:
            jobs_list = []
    return render_template("index.html", roles=roles, jobs=jobs_list)

@app.route("/analyze", methods=["POST"])
def analyze():
    if "resume" not in request.files:
        return "No resume file part in request. Try again.", 400
    file = request.files["resume"]
    if file.filename == "":
        return "No selected file", 400
    if not allowed_file(file.filename):
        return "Only PDF allowed", 400

    user_jd = request.form.get("job_description", "").strip()
    selected_role = request.form.get("selected_role", "").strip()

    # save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    raw_text = extract_text_from_pdf(filepath)
    resume_text = clean_text(raw_text)
    preview = resume_text[:2000] + ("..." if len(resume_text) > 2000 else "")

    # user JD based ATS
    user_ats = None
    user_matched = []
    user_missing = []

    if user_jd:
        jd_clean = clean_text(user_jd)
        user_ats = calculate_ats_score(resume_text, jd_clean)
        jd_keywords = extract_top_keywords(jd_clean, top_n=25)
        user_matched, user_missing = match_keywords(jd_keywords, resume_text)

    # selected role suitability
    role_score = None
    role_matched = []
    role_missing = []
    role_label = None
    if selected_role:
        role_score, role_matched, role_missing = role_suitability_score(selected_role, resume_text)
        if role_score is not None:
            if role_score >= 60: role_label = "Highly suitable"
            elif role_score >= 40: role_label = "Moderately suitable"
            else: role_label = "Not suitable"

    # sample jobs matching (multi job profiles)
    job_matches = []
    jobs_path = os.path.join("data", "jobs.csv")
    if os.path.exists(jobs_path):
        try:
            dfj = pd.read_csv(jobs_path)
            for _, r in dfj.iterrows():
                title = str(r.get("title",""))
                desc = str(r.get("description","")) if "description" in r else str(r.get("skills",""))
                desc_clean = clean_text(desc)
                score = calculate_ats_score(resume_text, desc_clean)
                # derive keywords for this job (prefer skills column if semicolon separated)
                if "skills" in r and pd.notna(r.get("skills")) and ";" in str(r.get("skills")):
                    job_kw = [s.strip() for s in str(r.get("skills")).split(";") if s.strip()]
                else:
                    job_kw = extract_top_keywords(desc_clean, top_n=20)
                matched, missing = match_keywords(job_kw, resume_text)
                job_matches.append({
                    "title": title,
                    "description": desc,
                    "score": round(score,2),
                    "matched": matched,
                    "missing": missing,
                    "matched_count": len(matched),
                    "missing_count": len(missing)
                })
            job_matches = sorted(job_matches, key=lambda x: x["score"], reverse=True)
        except Exception:
            job_matches = []

    # if no user_jd and no selected_role, infer from top job match if exists
    if not user_jd and not selected_role and job_matches:
        top = job_matches[0]
        user_ats = round(top.get("score", 0.0), 2)
        user_matched = top.get("matched", [])
        user_missing = top.get("missing", [])

    # Keyword optimization tips (sample phrases)
    optimization_tips = []
    for kw in user_missing:
        optimization_tips.append(f"Add a line describing experience with '{kw}' in Projects or Skills (e.g., 'Implemented X using {kw}').")

    # Multiple job top-5 lists for chart
    top_jobs = job_matches[:5]
    top_job_titles = [j["title"] for j in top_jobs]
    top_job_scores = [j["score"] for j in top_jobs]

    # Trend logging: choose role_label for history key
    history_key = selected_role if selected_role else (top_jobs[0]['title'] if top_jobs else "general")
    history_score = user_ats if user_ats is not None else (role_score if role_score is not None else (top_jobs[0]['score'] if top_jobs else 0.0))
    # append to history
    append_history(history_key, history_score)
    # read trend for this key
    trend_labels, trend_scores = get_trend_for_role(history_key, max_points=30)

    # prepare chart payload
    chart_payload = {
        "ats_score": user_ats if user_ats is not None else (role_score if role_score is not None else 0.0),
        "matched_count": len(user_matched) if user_matched else len(role_matched),
        "missing_count": len(user_missing) if user_missing else len(role_missing),
        "matched_keywords": user_matched if user_matched else role_matched,
        "missing_keywords": user_missing if user_missing else role_missing,
        "top_job_titles": top_job_titles,
        "top_job_scores": top_job_scores,
        "trend_labels": trend_labels,
        "trend_scores": trend_scores
    }

    # summary
    summary_text = ""
    if selected_role:
        summary_text = f"Suitability for role '{selected_role}': {role_score}% — {role_label}."
        if role_missing:
            summary_text += f" Add keywords: {', '.join(role_missing[:6])}."
    elif user_ats is not None:
        lab = "Low"
        if user_ats >= 80: lab = "Excellent"
        elif user_ats >= 50: lab = "Moderate"
        summary_text = f"ATS score for provided JD: {user_ats}% — {lab} match."
        if user_missing:
            summary_text += f" Add: {', '.join(user_missing[:6])}."
    elif job_matches:
        best = job_matches[0]
        summary_text = f"Top matched sample job: {best['title']} (score {best['score']}%)."

    suggestions = suggest_for_missing(user_missing if user_missing else role_missing)

    return render_template(
        "result.html",
        ats_score=chart_payload["ats_score"],
        preview=preview,
        matched_keywords=chart_payload["matched_keywords"],
        missing_keywords=chart_payload["missing_keywords"],
        chart_data=json.dumps(chart_payload),
        suggestions=suggestions,
        summary_text=summary_text,
        job_matches=job_matches,
        selected_role=selected_role,
        role_score=role_score,
        role_label=role_label,
        role_matched=role_matched,
        role_missing=role_missing,
        optimization_tips=optimization_tips
    )

# ---------- Run ----------
if __name__ == "__main__":
    app.run(debug=True)
