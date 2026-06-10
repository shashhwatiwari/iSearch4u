from typing import TypedDict
from langgraph.graph import StateGraph, END
from google import genai
import os
from dotenv import load_dotenv

load_dotenv(".env")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 1. State — the object that flows through every node 
class ResearchState(TypedDict):
    question:         str            # set at the start, never changes
    retrieved_chunks: list[str]      # filled by Retriever (empty for now)
    answer:           str            # filled by Synthesizer

# 2. The Synthesizer node
def synthesizer(state: ResearchState) -> dict:
    question = state["question"]

    # For now, no retrieved chunks — the model answers from its own knowledge
    prompt = f"""Answer the following question as clearly as you can.

Question: {question}

Answer:"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    # Return only the fields this node updated
    return {"answer": response.text}

# 3. Build the graph 
def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("synthesizer", synthesizer)   # register the node
    graph.set_entry_point("synthesizer")         # start here
    graph.add_edge("synthesizer", END)           # after this, we're done

    return graph.compile()


#  4. Run it 
if __name__ == "__main__":
    app = build_graph()

    result = app.invoke({
        "question": "What is the attention mechanism?",
        "retrieved_chunks": [],
        "answer": ""
    })

    print(result["answer"])