import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP
from agent.graphV4 import build_graph
# ── Create the MCP server ────────────────────────────────────────────────────
mcp = FastMCP("iSearch4u")
app = build_graph()

# ── Register your pipeline as a tool ────────────────────────────────────────
@mcp.tool()
def research(question: str) -> str:
    """Research a question using the local document collection."""
    result = app.invoke({
        "question":         question,
        "sub_queries":      [],
        "retrieved_chunks": [],
        "answer":           ""
    })
    return result["answer"]

# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run()