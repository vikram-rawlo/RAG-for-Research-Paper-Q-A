# RAG PDF Chatbot

A simple RAG (Retrieval-Augmented Generation) chatbot that answers questions from PDF documents using Streamlit, LangChain, and ChromaDB.

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your PDFs**
   - Place PDF files in `data/pdfs/` folder

3. **Set API key**
   - Create `.env` file
   - Add: `OPENAI_API_KEY=your_key_here`

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

## How it works

- **First run**: Processes PDFs and creates embeddings (takes time)
- **Subsequent runs**: Loads existing embeddings (fast startup)
- Ask questions about your PDF content in the chat interface

## Project Structure

```
rag-project/
├── app.py                    # Main Streamlit app
├── src/                      # Core components
├── config/settings.py        # Configuration
├── data/pdfs/               # Your PDF files
├── storage/chroma_db/       # Vector database (auto-created)
└── utils/helpers.py         # Utility functions
```

## Requirements

- Python 3.8+
- OpenAI API key
- PDF documents in `data/pdfs/` folder