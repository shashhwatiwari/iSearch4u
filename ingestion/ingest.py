# ingestion/ingest.py

import os
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer



# ── Config 
DOCS_DIR = Path("docs")          # where .md files live
DB_DIR   = "chroma_db"           # where ChromaDB will save to disk
CHUNK_SIZE    = 500              # characters per chunk (not tokens)
CHUNK_OVERLAP = 50               #  overlap between chunks




# ── 1. Load the embedding model (runs locally, no API key) ───────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")




# ── 2. Connect to ChromaDB (creates the folder if it doesn't exist) ──────────
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




# ── 4. Ingest one file ───────────────────────────────────────────────────────
def ingest_file(filepath: Path):
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
    md_files = list(DOCS_DIR.glob("*.md"))

    if not md_files:
        print("No .md files found in docs/. Add some and try again.")
    else:
        for filepath in md_files:
            print(f"\nIngesting: {filepath.name}")
            ingest_file(filepath)

    print(f"\nDone. {collection.count()} chunks stored in ChromaDB.")