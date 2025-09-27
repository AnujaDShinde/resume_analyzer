# modules/parser.py
import fitz  # PyMuPDF
import docx2txt
import os

def extract_text_from_pdf(path):
    text = ""
    doc = fitz.open(path)
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(path):
    try:
        return docx2txt.process(path) or ""
    except Exception:
        return ""

def extract_text(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext in [".docx", ".doc"]:
        return extract_text_from_docx(path)
    elif ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError("Unsupported file type: " + ext)

if __name__ == "__main__":
    # quick test (adjust the path)
    p = "data/sample_resumes/example_resume.pdf"
    if os.path.exists(p):
        print(extract_text(p)[:1000])
    else:
        print("Place a sample resume at:", p)
