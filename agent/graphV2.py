# This file defines a simple graph-based agent that answers research questions by retrieving relevant document 
# chunks and synthesizing an answer using Gemini. It uses langgraph to define the graph structure, 
# chromadb for vector storage, and sentence-transformers for embeddings.

from typing import TypedDict
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
import os
from dotenv import load_dotenv

load_dotenv(".env")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Shared resources (loaded once at startup) 
embedder       = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client  = chromadb.PersistentClient(path="chroma_db")
collection     = chroma_client.get_collection(name="documents")

#  1. State
class ResearchState(TypedDict):
    question:         str
    retrieved_chunks: list[str]
    answer:           str

#  2. Retriever node
def retriever(state: ResearchState) -> dict:
    question  = state["question"]
    embedding = embedder.encode(question).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=5
    )

    chunks = results["documents"][0]   # list of 5 raw text chunks
    print(f"[Retriever] Found {len(chunks)} chunks")
    return {"retrieved_chunks": chunks}

#  3. Synthesizer node
def synthesizer(state: ResearchState) -> dict:
    question = state["question"]
    chunks   = state["retrieved_chunks"]

    # Build context string from retrieved chunks
    context = "\n\n---\n\n".join(chunks)

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

#  4. Build the graph
def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("retriever",   retriever)
    graph.add_node("synthesizer", synthesizer)

    graph.set_entry_point("retriever")          # start here
    graph.add_edge("retriever",   "synthesizer") # then go here
    graph.add_edge("synthesizer", END)           # then done

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    result = app.invoke({
        "question": "What is the attention mechanism?",
        "retrieved_chunks": [],
        "answer": ""
    })

    print(result["answer"])