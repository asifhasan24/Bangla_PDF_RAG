import re
import fitz  # PyMuPDF

def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    full_text = "\n".join(pages)
    return re.sub(r'\s+', ' ', full_text).strip()
