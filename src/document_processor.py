import os

from typing import List
import pypdf
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config.settings import PDF_DIRECTORY, CHUNK_SIZE, CHUNK_OVERLAP
from utils.helpers import log_message, clean_text


class DocumentProcessor:
    """
    Handles loading and processing of PDF documents into clean, chunked text.
    """
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        log_message(f"DocumentProcessor initialized with chunk_size={self.chunk_size}, "
                    f"chunk_overlap={self.chunk_overlap}")

    def load_pdf(self, file_path: str) -> List[Document]:
        """Load text content from a single PDF file as page-level Documents."""
        documents = []
        try:
            reader = pypdf.PdfReader(file_path)
            for i, page in enumerate(reader.pages):
                page_content = page.extract_text()
                if page_content:
                    cleaned_content = clean_text(page_content)
                    documents.append(
                        Document(
                            page_content=cleaned_content,
                            metadata={"source": os.path.basename(file_path), "page": i + 1}
                        )
                    )
            log_message(f"Loaded {len(documents)} pages from {file_path}")
        except Exception as e:
            log_message(f"Error loading PDF {file_path}: {str(e)}", level="error")
        return documents

    def load_all_pdfs(self, directory: str = PDF_DIRECTORY) -> List[Document]:
        """Load text content from all PDF files in a directory."""
        all_documents = []
        if not os.path.exists(directory):
            log_message(f"PDF directory not found: {directory}", level="error")
            return []

        pdf_files = [f for f in os.listdir(directory) if f.lower().endswith(".pdf")]
        if not pdf_files:
            log_message(f"No PDF files found in directory: {directory}", level="warning")
            return []

        for pdf_file in pdf_files:
            file_path = os.path.join(directory, pdf_file)
            docs = self.load_pdf(file_path)
            all_documents.extend(docs)

        log_message(f"Total pages loaded from all PDFs: {len(all_documents)}")
        return all_documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller, overlapping chunks."""
        if not documents:
            return []

        chunks = self.text_splitter.split_documents(documents)
        log_message(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    def process_all_pdfs(self, directory: str = PDF_DIRECTORY) -> List[Document]:
        """Load all PDFs and return ready-to-use text chunks."""
        documents = self.load_all_pdfs(directory)
        if not documents:
            return []
        return self.split_documents(documents)