import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from memory import ChatMemory
from retriever import Retriever
from tasks import generate_answer_task, process_document_task
from celery.result import AsyncResult
from config import DEFAULT_INDEX_PATH, DEFAULT_METADATA_PATH, UPLOAD_DIR

# 1️⃣ Load env
load_dotenv()

# 2️⃣ Initialize FastAPI & components
app = FastAPI(
    title="RAG Chat API",
    description="Accepts English & Bangla queries, maintains short-term memory, retrieves long-term docs, and returns grounded answers."
)

memory = ChatMemory(max_messages=10)
retriever = Retriever(
    index_path=DEFAULT_INDEX_PATH,
    metadata_path=DEFAULT_METADATA_PATH
)

# 3️⃣ Request/Response schemas
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    task_id: str

class ResultResponse(BaseModel):
    status: str
    answer: str = None

class UploadResponse(BaseModel):
    task_id: str

class StatusResponse(BaseModel):
    status: str
    info: str = None

# 4️⃣ Endpoints
@app.post("/upload", response_model=UploadResponse)
async def upload_endpoint(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    task = process_document_task.delay(file_path)
    return UploadResponse(task_id=task.id)

@app.get("/ingest_status/{task_id}", response_model=StatusResponse)
async def ingest_status_endpoint(task_id: str):
    task = AsyncResult(task_id)
    if not task.ready():
        return StatusResponse(status=task.status)
    
    result = task.get()
    return StatusResponse(status=task.status, info=str(result))

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    q = req.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # short-term memory
    memory.add_user_message(q)

    # long-term retrieval
    docs = retriever.get_top_k(query=q, k=3)

    # generate
    ctx = memory.get_context()
    task = generate_answer_task.delay(question=q, context=ctx, documents=docs)

    return QueryResponse(task_id=task.id)

@app.get("/result/{task_id}", response_model=ResultResponse)
async def result_endpoint(task_id: str):
    task = AsyncResult(task_id)
    if not task.ready():
        return ResultResponse(status=task.status)

    result = task.get()
    memory.add_bot_message(result)
    return ResultResponse(status=task.status, answer=result)
