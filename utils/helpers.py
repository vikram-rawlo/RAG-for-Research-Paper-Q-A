import logging
import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

def setup_logging():
    """Setup basic logging configuration (safe against duplicate handlers)."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "rag_app.log"),
            logging.StreamHandler()
        ]
    )

def log_message(message: str, level: str = "info"):
    """Log a message with timestamp."""
    logger = logging.getLogger(__name__)
    getattr(logger, level.lower(), logger.info)(message)

def validate_pdf_files(pdf_folder: str) -> List[str]:
    """Validate that PDF files exist in the specified folder."""
    pdf_path = Path(pdf_folder)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF folder not found: {pdf_folder}")
    pdf_files = list(pdf_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdf_folder}")
    return [str(pdf) for pdf in pdf_files]

def ensure_directory_exists(directory: str):
    """Ensure a directory exists, create if it doesn't."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def initialize_app_directories():
    """Initialize required directories for the application."""
    for directory in ["data/pdfs", "storage/chroma_db", "logs"]:
        ensure_directory_exists(directory)
    setup_logging()
    log_message("Application directories initialized")

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and null characters."""
    if not text:
        return ""
    cleaned = " ".join(text.split())
    return cleaned.replace("\x00", "")

def format_response(response: str) -> str:
    """Format the AI response for display in Streamlit."""
    if not response:
        return "I couldn't find relevant information to answer your question."
    return response.strip()

def calculate_document_hash(text: str) -> str:
    """Generate a unique hash ID for a document chunk."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_all_pdfs(pdf_directory: str) -> List[Document]:
    """
    Load all PDFs from the given directory into LangChain Document objects.

    Args:
        pdf_directory (str): Path to the folder containing PDF files.

    Returns:
        List[Document]: List of LangChain Document objects.
    """
    documents = []
    if not os.path.exists(pdf_directory):
        raise FileNotFoundError(f"PDF directory not found: {pdf_directory}")

    for file_name in os.listdir(pdf_directory):
        if file_name.lower().endswith(".pdf"):
            file_path = os.path.join(pdf_directory, file_name)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)

    return documents