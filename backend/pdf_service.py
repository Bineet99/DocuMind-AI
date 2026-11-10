
import fitz  # PyMuPDF
import os

from backend.config import UPLOAD_FOLDER, MAX_FILE_SIZE


def save_pdf(file):
    """
    Save uploaded PDF to the data folder.
    """

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    filename = os.path.basename(file.filename)

    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are allowed.")

    file_path = os.path.join(UPLOAD_FOLDER, filename)

    # Save uploaded file
    with open(file_path, "wb") as buffer:
        content = file.file.read(MAX_FILE_SIZE + 1)

        if len(content) > MAX_FILE_SIZE:
            raise ValueError("PDF size exceeds 10 MB limit.")

        buffer.write(content)

    return file_path


def extract_text(file_path):
    """
    Extract text from PDF page by page.
    """

    pages = []

    try:
        document = fitz.open(file_path)

        if document.is_encrypted:
            raise ValueError("Password-protected PDFs are not supported.")

        for page_number, page in enumerate(document, start=1):

            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "page": page_number,
                    "content": text
                })

        document.close()

        return pages

    except Exception as e:
        raise ValueError(f"PDF extraction failed: {str(e)}")


def get_pdf_info(file_path):
    """
    Return basic PDF information.
    """

    document = fitz.open(file_path)

    info = {
        "filename": os.path.basename(file_path),
        "total_pages": len(document),
        "file_size": os.path.getsize(file_path)
    }

    document.close()

    return info

