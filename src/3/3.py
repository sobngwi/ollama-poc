"""Categorize and sort the grocery list in data/ with a local Ollama model.

Usage: python3 3.py [--host URL] [--model MODEL] [--input FILE] [--output FILE]
"""

from grocery import main

if __name__ == "__main__":
    raise SystemExit(main())
