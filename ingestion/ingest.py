# ingestion/ingest.py

import os
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Config ────────────────────────────────────────────────────────────────────
DOCS_DIR      = Path(PROJECT_ROOT) / "docs"
DB_DIR        = os.path.join(PROJECT_ROOT, "chroma_db")
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

# ── 1. Load the embedding model ───────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── 2. Connect to ChromaDB ────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_or_create_collection(name="documents")




# ── 3. Chunking function ─────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap   # step forward, but overlap a little
    return chunks


from pypdf import PdfReader

def read_pdf(filepath: Path) -> str:
    reader = PdfReader(filepath)
    # Extract text from every page and join with newlines
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# ── 4. Ingest one file ───────────────────────────────────────────────────────
def ingest_file(filepath: Path):
    if filepath.suffix == ".pdf":
        text = read_pdf(filepath)
    else:
        text = filepath.read_text(encoding="utf-8")
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

    for i, chunk in enumerate(chunks):
        embedding = embedder.encode(chunk).tolist()   # list of floats

        collection.add(
            ids=[f"{filepath.name}__chunk{i}"],       # unique ID for this chunk
            embeddings=[embedding],
            documents=[chunk],                         # the raw text
            metadatas=[{"source": filepath.name, "chunk_index": i}]
        )
        print(f"  Stored: {filepath.name} chunk {i}")




# ── 5. Main — loop over all .md files in the docs folder ────────────────────
if __name__ == "__main__":
    files = list(DOCS_DIR.glob("*.md")) + list(DOCS_DIR.glob("*.pdf"))

    if not files:
        print("No .md files found in docs/. Add some and try again.")
    else:
        for filepath in files:
            print(f"\nIngesting: {filepath.name}")
            ingest_file(filepath)

    print(f"\nDone. {collection.count()} chunks stored in ChromaDB.")

    

def ingest_file_to_collection(filepath: Path, collection_name: str) -> int:
    """Ingest a single file into a named ChromaDB collection. Returns chunk count."""
    col = chroma_client.get_or_create_collection(name=collection_name)
    
    if filepath.suffix == ".pdf":
        text = read_pdf(filepath)
    else:
        text = filepath.read_text(encoding="utf-8")

    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

    for i, chunk in enumerate(chunks):
        embedding = embedder.encode(chunk).tolist()
        col.add(
            ids=[f"{filepath.name}__chunk{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": filepath.name, "chunk_index": i}]
        )

    return len(chunks)