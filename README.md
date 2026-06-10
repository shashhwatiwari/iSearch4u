# iSearch4u — Multi-Agent Research Assistant

A local research assistant that ingests your documents, stores them in a vector database, and answers questions using a multi-agent LangGraph pipeline. Callable from Claude Desktop via MCP.

---

## What it does

1. You drop `.md` files into a `docs/` folder and run the ingestion script
2. Documents are chunked, embedded, and stored persistently in ChromaDB
3. When you ask a question, a 4-node agent pipeline runs:
   - **Planner** — breaks your question into focused sub-queries
   - **Retriever** — searches ChromaDB for relevant chunks
   - **Synthesizer** — writes a grounded answer using only the retrieved context
   - **Critic** — evaluates the answer and loops back to Retriever if quality is poor
4. The whole pipeline is exposed as an MCP tool, callable from Claude Desktop

---

## Tech stack

| Component | Tool |
|---|---|
| Language | Python 3.10+ |
| LLM | Gemini API (google-genai) |
| Agent orchestration | LangGraph |
| Vector database | ChromaDB (local, persistent) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Server interface | MCP (Model Context Protocol) |

---

## Project structure

```
iSearch4u/
├── agent/
│   ├── graphV1.py        # Synthesizer only
│   ├── graphV2.py        # Retriever + Synthesizer
│   ├── graphV3.py        # Planner + Retriever + Synthesizer
│   └── graphV4.py        # + Critic with conditional loop (current)
├── ingestion/
│   ├── ingest.py         # Chunk, embed, and store documents
│   └── query.py          # Test semantic search directly
├── StarterKit/
│   ├── simple_call.py    # Basic Gemini API call
│   ├── looped_call.py    # Multi-turn chat loop
│   └── tool_agent.py     # Tool-use agent loop from scratch
├── docs/                 # Drop your .md files here
├── mcp_server.py         # MCP server — wraps the pipeline as a tool
├── .env.example          # Environment variable template
└── .gitignore
```

---

## Setup

**1. Clone the repo and install dependencies**

```bash
pip install mcp chromadb sentence-transformers langgraph google-genai python-dotenv
```

**2. Add your Gemini API key**

```bash
cp .env.example .env
# then open .env and add your key
```

**3. Add documents**

Drop `.md` files into the `docs/` folder.

**4. Ingest documents**

```bash
python ingestion/ingest.py
```

**5. Test retrieval**

```bash
python ingestion/query.py
```

**6. Run the pipeline directly**

```bash
python agent/graphV4.py
```

---

## Claude Desktop setup (MCP)

Add this to your Claude Desktop config at `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "iSearch4u": {
      "command": "/path/to/your/python3",
      "args": ["/absolute/path/to/iSearch4u/mcp_server.py"],
      "cwd": "/absolute/path/to/iSearch4u"
    }
  }
}
```

Find your Python path with `which python3`. Restart Claude Desktop after saving.

---

## How the agent graph works

```
User question
      ↓
  [ Planner ]   — asks LLM to generate 1-3 focused sub-queries
      ↓
  [ Retriever ] — embeds each sub-query, searches ChromaDB, deduplicates chunks
      ↓
  [ Synthesizer ] — answers using only retrieved context
      ↓
  [ Critic ]    — verdict: GOOD → done | BAD → loop back to Retriever (max 3 iterations)
      ↓
   Answer
```

State flows through every node as a typed dictionary — each node reads what it needs and writes back only what it produced.

---

## Environment variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Your Gemini API key from Google AI Studio |
