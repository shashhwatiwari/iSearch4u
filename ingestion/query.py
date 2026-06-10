# ingestion/query.py

import chromadb
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_collection(name="documents")

def search(question: str, top_k: int = 3):
    # Embed the question the same way we embedded the chunks
    question_embedding = embedder.encode(question).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    print(f"\nQuery: {question}\n")
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"── Result {i+1} ──────────────────────────")
        print(f"Source: {meta['source']}  |  Chunk: {meta['chunk_index']}")
        print(doc[:300])   # first 300 chars so it's readable
        print()

if __name__ == "__main__":
    search("What is the attention mechanism?")
    search("How does multi-head attention work?")