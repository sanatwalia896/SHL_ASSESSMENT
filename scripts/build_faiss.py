from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.vector_store import get_vector_store

if __name__ == "__main__":
    get_vector_store()
    print("FAISS index built successfully.")