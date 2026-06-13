# backend/main.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import tempfile, uuid, shutil

from ingestion.ingest import ingest_file_to_collection
from agent.graphV4 import build_graph_for_collection

app = FastAPI()

# Allow requests from your frontend (Vercel or localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request/response models ───────────────────────────────────────────────────
class AskRequest(BaseModel):
    session_id: str
    question:   str

class AskResponse(BaseModel):
    answer:     str
    session_id: str

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    # Generate a unique session ID for this upload
    session_id = str(uuid.uuid4())

    # Save the uploaded file to a temp location
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    # Ingest into a session-specific ChromaDB collection
    chunk_count = ingest_file_to_collection(tmp_path, collection_name=session_id)
    tmp_path.unlink()   # delete the temp file

    return {"session_id": session_id, "chunks_stored": chunk_count}

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    graph = build_graph_for_collection(request.session_id)

    result = graph.invoke({
        "question":         request.question,
        "sub_queries":      [],
        "retrieved_chunks": [],
        "answer":           "",
        "iteration_count":  0,
        "critique":         ""
    })

    return AskResponse(answer=result["answer"], session_id=request.session_id)