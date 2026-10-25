"""Ask a local Ollama model for a short funny story and stream it to the terminal.

Usage: python3 1.py [--url URL] [--model MODEL] [--prompt PROMPT]
"""

from story import main

if __name__ == "__main__":
    raise SystemExit(main())
