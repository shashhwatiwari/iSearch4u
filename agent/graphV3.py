# agent/graph.py

from typing import TypedDict
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
import os, json
from dotenv import load_dotenv

# Absolute path to project root — works regardless of where the script is launched from
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── Shared resources ─────────────────────────────────────────────────────────
embedder      = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=os.path.join(PROJECT_ROOT, "chroma_db"))
collection    = chroma_client.get_collection(name="documents")

# ── 1. State ──────────────────────────────────────────────────────────────────
class ResearchState(TypedDict):
    question:         str
    sub_queries:      list[str]   # NEW — filled by Planner
    retrieved_chunks: list[str]
    answer:           str

# ── 2. Planner node ───────────────────────────────────────────────────────────
def planner(state: ResearchState) -> dict:
    question = state["question"]

    prompt = f"""You are a research planner. Break the question below into 1-3 focused 
sub-queries that can each be searched independently in a document database.
Return ONLY a JSON array of strings. No explanation, no markdown, just the array.

Question: {question}

Example output: ["sub-query one", "sub-query two"]"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    sub_queries = json.loads(response.text)
    import sys; #   print(f"[Planner] Sub-queries: {sub_queries}", file=sys.stderr)
    return {"sub_queries": sub_queries}

# ── 3. Retriever node ─────────────────────────────────────────────────────────
def retriever(state: ResearchState) -> dict:
    sub_queries = state["sub_queries"]
    all_chunks  = []
    seen        = set()   # for deduplication

    for query in sub_queries:
        embedding = embedder.encode(query).tolist()
        results   = collection.query(query_embeddings=[embedding], n_results=3)

        for chunk in results["documents"][0]:
            if chunk not in seen:
                seen.add(chunk)
                all_chunks.append(chunk)

    import sys; # print(f"[Retriever] Retrieved {len(all_chunks)} unique chunks", file=sys.stderr)
    return {"retrieved_chunks": all_chunks}

# ── 4. Synthesizer node ───────────────────────────────────────────────────────
def synthesizer(state: ResearchState) -> dict:
    question = state["question"]
    chunks   = state["retrieved_chunks"]
    context  = "\n\n---\n\n".join(chunks)

    prompt = f"""You are a research assistant. Answer the question using ONLY the context below.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return {"answer": response.text}

# ── 5. Build the graph ────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("planner",     planner)
    graph.add_node("retriever",   retriever)
    graph.add_node("synthesizer", synthesizer)

    graph.set_entry_point("planner")
    graph.add_edge("planner",     "retriever")
    graph.add_edge("retriever",   "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()

# ── 6. Run it ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = build_graph()

    result = app.invoke({
        "question": "How does the Transformer compare to RNNs in terms of computational complexity?",
        "sub_queries":      [],
        "retrieved_chunks": [],
        "answer":           ""
    })

    print("\n" + result["answer"])