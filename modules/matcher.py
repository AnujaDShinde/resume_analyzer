# modules/matcher.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# model download first time (internet required)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed_texts(texts):
    model = get_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

def match_resume_to_jobs(resume_text, jobs_df, top_n=5):
    # jobs_df must have columns: title, description, skills
    corpus = (jobs_df['title'].fillna('') + " " + jobs_df['description'].fillna('')).tolist()
    corpus_embeddings = embed_texts(corpus)
    resume_embedding = embed_texts([resume_text])[0]
    sims = cosine_similarity([resume_embedding], corpus_embeddings)[0]
    jobs_df = jobs_df.copy()
    jobs_df['score'] = sims
    top = jobs_df.sort_values('score', ascending=False).head(top_n)
    # round score
    top['score'] = (top['score'] * 100).round(2)
    return top.reset_index(drop=True)

def missing_skills(resume_skills, job_skills_str):
    # job_skills_str is semicolon separated
    job_skills = [s.strip().lower() for s in str(job_skills_str).split(';') if s.strip()]
    rset = set([s.lower() for s in resume_skills])
    missing = [s for s in job_skills if s not in rset]
    return missing

if __name__ == "__main__":
    # quick local test
    df = pd.DataFrame([
        {"title":"Data Scientist", "description":"Build ML models and EDA", "skills":"Python;Pandas;scikit-learn;SQL"},
        {"title":"ML Engineer", "description":"Deploy models with Docker and AWS", "skills":"Python;TensorFlow;Docker;AWS"}
    ])
    res = match_resume_to_jobs("I have experience with Python, pandas and SQL. Built models in sklearn.", df, top_n=2)
    print(res)
    print(missing_skills(["python","pandas","sql"], res.loc[0,"skills"]))
