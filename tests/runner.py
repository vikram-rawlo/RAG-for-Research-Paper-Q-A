import os
import sys
import logging

# Ensure project root is in sys.path (important when running from tests/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# Import project modules
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore
from src.qa_chain import ask_question
from utils.helpers import load_all_pdfs, log_message
from config import settings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_tests():
    print("\n🚀 Running integration tests...\n")

    # Step 1: Initialize embeddings
    try:
        embedding_manager = EmbeddingManager()
        embeddings = embedding_manager.get_embeddings_model()
        print("✅ EmbeddingManager initialized.")
    except Exception as e:
        print(f"❌ Failed to initialize EmbeddingManager: {e}")
        return

    # Step 2: Initialize vector store
    try:
        vector_store = VectorStore(embeddings)
        print("✅ VectorStore initialized.")
    except Exception as e:
        print(f"❌ Failed to initialize VectorStore: {e}")
        return

    # Step 3: Load sample documents (put test PDFs in tests/data/)
    try:
        test_data_dir = os.path.join(ROOT_DIR, "tests", "data")
        docs = load_all_pdfs(test_data_dir)
        if not docs:
            print("⚠️ No test documents found in tests/data/. Please add a small PDF.")
            return
        print(f"✅ Loaded {len(docs)} documents from {test_data_dir}.")
    except Exception as e:
        print(f"❌ Failed to load test documents: {e}")
        return

    # Step 4: Add documents to vector store
    try:
        vector_store.add_documents(docs)
        print("✅ Documents added to vector store.")
    except Exception as e:
        print(f"❌ Failed to add documents: {e}")
        return

    # Step 5: Run retriever test
    try:
        retriever = vector_store.get_retriever(k=2)
        results = retriever.invoke("test query")
        print(f"✅ Retriever returned {len(results)} documents.")
    except Exception as e:
        print(f"❌ Retriever test failed: {e}")
        return

    # Step 6: Run QA chain
    try:
        response = ask_question("What is Language model meta-learning?")
        print("✅ QA chain executed successfully.")
        print("--- Answer ---")
        print(response["answer"])
        print("--- Metadata ---")
        print(f"Confidence: {response['confidence']}, Chunks: {response['retrieved_chunks']}")
    except Exception as e:
        print(f"❌ QA chain test failed: {e}")
        return

    print("\n🎉 All integration tests completed successfully!\n")


if __name__ == "__main__":
    run_tests()
