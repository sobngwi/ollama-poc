import sys
from pathlib import Path

# Make this scenario's modules (pdf_rag.py) importable from its tests.
sys.path.insert(0, str(Path(__file__).resolve().parent))
