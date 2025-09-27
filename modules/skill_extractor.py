# modules/skill_extractor.py
import re
import spacy

nlp = spacy.load("en_core_web_sm")

# A basic skill list — तुम्ही ह्याला वाढवू शकता
SKILLS = {
    "python","java","c++","c","sql","mongodb","mysql","postgresql",
    "pandas","numpy","scikit-learn","tensorflow","keras","pytorch",
    "docker","kubernetes","aws","azure","gcp","linux","git",
    "html","css","javascript","react","nodejs","tableau","powerbi",
    "spark","hadoop","nlp","computer vision","opencv","matplotlib",
    "seaborn","rest api","flask","django","fastapi","data analysis",
    "machine learning","deep learning","reinforcement learning"
}

# Normalize text: lowercase and remove extra spaces
def normalize(text):
    return re.sub(r'\s+', ' ', text).strip().lower()

def extract_skills(text):
    text_norm = normalize(text)
    found = set()

    # direct keyword matching
    for skill in SKILLS:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_norm):
            found.add(skill)

    # use spaCy to catch some multi-word noun chunks (like "machine learning")
    doc = nlp(text)
    for chunk in doc.noun_chunks:
        ch = chunk.text.lower().strip()
        if ch in SKILLS:
            found.add(ch)

    # return sorted list
    return sorted(found)

if __name__ == "__main__":
    sample = "Experienced in Python, pandas, SQL and deployed models using Docker on AWS."
    print(extract_skills(sample))
