import os
import re
from datetime import datetime
from PIL import Image
import pytesseract

# ✅ Updated LangChain imports
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader
)

from langchain_core.documents import Document


# -------- CLEAN TEXT --------
def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""

    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    text = re.sub(r"[^\w\s.,!?;:()\-\'\"]", "", text)  # remove unwanted chars
    return text.strip()


# -------- METADATA --------
def extract_metadata(file_path: str) -> dict:
    """Extract metadata from file."""
    try:
        return {
            "source": os.path.basename(file_path),
            "file_size": os.path.getsize(file_path),
            "last_modified": datetime.fromtimestamp(
                os.path.getmtime(file_path)
            ).isoformat(),
            "file_type": os.path.splitext(file_path)[1].lower()
        }
    except Exception:
        return {
            "source": os.path.basename(file_path),
            "file_type": os.path.splitext(file_path)[1].lower()
        }


# -------- PROCESS DOCS --------
def process_docs(docs, metadata):
    """Clean content and attach metadata."""
    processed = []
    for doc in docs:
        if doc.page_content:
            doc.page_content = clean_text(doc.page_content)
            doc.metadata.update(metadata)
            processed.append(doc)
    return processed


# -------- FILE LOADERS --------
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return process_docs(loader.load(), extract_metadata(file_path))


def load_text(file_path):
    loader = TextLoader(file_path, encoding="utf-8")
    return process_docs(loader.load(), extract_metadata(file_path))


def load_word(file_path):
    loader = UnstructuredWordDocumentLoader(file_path)
    return process_docs(loader.load(), extract_metadata(file_path))


def load_excel(file_path):
    loader = UnstructuredExcelLoader(file_path)
    return process_docs(loader.load(), extract_metadata(file_path))


def load_powerpoint(file_path):
    loader = UnstructuredPowerPointLoader(file_path)
    return process_docs(loader.load(), extract_metadata(file_path))


def load_html(file_path):
    loader = UnstructuredHTMLLoader(file_path)
    return process_docs(loader.load(), extract_metadata(file_path))


def load_markdown(file_path):
    loader = UnstructuredMarkdownLoader(file_path)
    return process_docs(loader.load(), extract_metadata(file_path))


def load_image(file_path):
    """Extract text from image using OCR."""
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
    except Exception:
        text = "OCR failed or Tesseract not installed"

    metadata = extract_metadata(file_path)
    return [Document(page_content=clean_text(text), metadata=metadata)]


# -------- MAIN ENTRY --------
def load_file(file_path):
    """Load file based on extension."""
    ext = os.path.splitext(file_path)[1].lower()

    loaders = {
        ".pdf": load_pdf,
        ".txt": load_text,
        ".docx": load_word,
        ".xlsx": load_excel,
        ".pptx": load_powerpoint,
        ".html": load_html,
        ".md": load_markdown,
        ".png": load_image,
        ".jpg": load_image,
        ".jpeg": load_image
    }

    if ext in loaders:
        return loaders[ext](file_path)

    raise ValueError(f"Unsupported file type: {ext}")
