import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# API Keys and Authentication
# =============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =============================================================================
# File Paths and Directories
# =============================================================================
PDF_DIRECTORY = "data/pdfs"

# Unified persistence directory name for ChromaDB
PERSIST_DIRECTORY = "storage/chroma_db"   # <- alias for backward compatibility
CHROMA_DB_DIRECTORY = PERSIST_DIRECTORY   # <- prefer using this going forward

# =============================================================================
# Document Processing Settings
# =============================================================================
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# =============================================================================
# Embedding Settings
# =============================================================================
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding model

# =============================================================================
# Vector Store Settings
# =============================================================================
COLLECTION_NAME = "pdf_documents"
SEARCH_RESULTS_COUNT = 3  # Number of relevant chunks to retrieve

# =============================================================================
# LLM Settings
# =============================================================================
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0

# =============================================================================
# Streamlit Settings
# =============================================================================
APP_TITLE = "PDF Q&A System"
APP_ICON = "📚"

# =============================================================================
# Retrieval Settings
# =============================================================================
MIN_SCORE_THRESHOLD = 0.3  # Minimum similarity score to consider a match relevant

# =============================================================================
# Embedding Workflow Settings
# =============================================================================
FORCE_REBUILD = False  # If True, always rebuild embeddings even if ChromaDB exists

# =============================================================================
# Validation
# =============================================================================
def validate_settings():
    """Validate that all required settings are properly configured"""
    errors = []
    
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY not found in environment variables")
    
    if not os.path.exists(PDF_DIRECTORY):
        errors.append(f"PDF directory not found: {PDF_DIRECTORY}")

    if not (0 <= MIN_SCORE_THRESHOLD <= 1):
        errors.append("MIN_SCORE_THRESHOLD must be between 0 and 1")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"- {error}" for error in errors))
    
    return True

# Validate settings on import
if __name__ != "__main__":
    validate_settings()