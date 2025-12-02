import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from memory import ChatMemory
from retriever import Retriever
from tasks import generate_answer_task
from celery.result import AsyncResult

# 1️⃣ Load env
load_dotenv()

# 2️⃣ Initialize FastAPI & components
app = FastAPI(
    title="RAG Chat API",
    description="Accepts English & Bangla queries, maintains short-term memory, retrieves long-term docs, and returns grounded answers."
)

memory = ChatMemory(max_messages=10)
retriever = Retriever(
    index_path="/home/asif/Documents/rag_project/vector_store/index.faiss",
    metadata_path="/home/asif/Documents/rag_project/vector_store/metadata.json"
)

# 3️⃣ Request/Response schemas
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    task_id: str

class ResultResponse(BaseModel):
    status: str
    answer: str = None

# 4️⃣ Endpoints
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
