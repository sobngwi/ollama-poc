"""Answer a question about data/BOI.pdf with a local RAG pipeline (LangChain, Chroma, Ollama).

Usage: python3 4.py [--pdf FILE] [--question TEXT] [--model MODEL]
                    [--embedding-model MODEL] [--host URL] [--trace]
"""

from pdf_rag import main

if __name__ == "__main__":
    raise SystemExit(main())
