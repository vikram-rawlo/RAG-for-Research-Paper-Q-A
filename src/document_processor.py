import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config.settings import PDF_DIRECTORY, CHUNK_SIZE, CHUNK_OVERLAP


class DocumentProcessor:
    """Handles loading and processing PDF documents"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
    
    def get_pdf_files(self) -> List[str]:
        """Get all PDF files from the PDF directory"""
        if not os.path.exists(PDF_DIRECTORY):
            raise FileNotFoundError(f"PDF directory not found: {PDF_DIRECTORY}")
        
        pdf_files = []
        for filename in os.listdir(PDF_DIRECTORY):
            if filename.lower().endswith('.pdf'):
                full_path = os.path.join(PDF_DIRECTORY, filename)
                pdf_files.append(full_path)
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {PDF_DIRECTORY}")
        
        print(f"Found {len(pdf_files)} PDF files:")
        for pdf_file in pdf_files:
            print(f"  - {os.path.basename(pdf_file)}")
        
        return pdf_files
    
    def load_single_pdf(self, pdf_path: str) -> List[Document]:
        """Load a single PDF file and return documents"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"Loading PDF: {os.path.basename(pdf_path)}")
        
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            print(f"  - Loaded {len(documents)} pages")
            return documents
        except Exception as e:
            print(f"  - Error loading PDF: {e}")
            return []
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks"""
        if not documents:
            return []
        
        print(f"Chunking {len(documents)} documents...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"  - Created {len(chunks)} chunks")
        
        return chunks
    
    def process_all_pdfs(self) -> List[Document]:
        """Process all PDF files and return chunked documents"""
        print("🚀 Processing all PDF files...")
        
        # Get all PDF files
        pdf_files = self.get_pdf_files()
        
        all_documents = []
        
        # Process each PDF
        for pdf_file in pdf_files:
            documents = self.load_single_pdf(pdf_file)
            if documents:
                chunks = self.chunk_documents(documents)
                all_documents.extend(chunks)
        
        print(f"✅ Total processing complete: {len(all_documents)} chunks from {len(pdf_files)} PDFs")
        return all_documents
    
    def get_document_stats(self, documents: List[Document]) -> dict:
        """Get statistics about the processed documents"""
        if not documents:
            return {"total_chunks": 0, "total_characters": 0, "average_chunk_size": 0}
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        avg_chunk_size = total_chars / len(documents) if documents else 0
        
        # Count unique sources
        sources = set()
        for doc in documents:
            if 'source' in doc.metadata:
                sources.add(os.path.basename(doc.metadata['source']))
        
        return {
            "total_chunks": len(documents),
            "total_characters": total_chars,
            "average_chunk_size": int(avg_chunk_size),
            "unique_sources": len(sources),
            "source_files": list(sources)
        }


# For testing the document processor
if __name__ == "__main__":
    processor = DocumentProcessor()
    
    try:
        # Process all PDFs
        documents = processor.process_all_pdfs()
        
        # Show statistics
        stats = processor.get_document_stats(documents)
        print(f"\n📊 Document Statistics:")
        print(f"  - Total chunks: {stats['total_chunks']}")
        print(f"  - Total characters: {stats['total_characters']:,}")
        print(f"  - Average chunk size: {stats['average_chunk_size']} characters")
        print(f"  - Source files: {stats['source_files']}")
        
        # Show first chunk as sample
        if documents:
            print(f"\n📄 Sample chunk (first 200 chars):")
            print(f"  {documents[0].page_content[:200]}...")
            print(f"  Metadata: {documents[0].metadata}")
        
    except Exception as e:
        print(f"❌ Error: {e}")