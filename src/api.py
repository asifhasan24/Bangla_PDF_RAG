import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from .memory   import ChatMemory
from .retriever import Retriever
# from .generator import Generator
from .generator import GeminiGenerator

# 1️⃣ Load env
load_dotenv()

# 2️⃣ Initialize FastAPI & components
app = FastAPI(
    title="RAG Chat API",
    description="Accepts English & Bangla queries, maintains short-term memory, retrieves long-term docs, and returns grounded answers."
)

memory = ChatMemory(max_messages=10)
retriever = Retriever(
    index_path="vector_store/index.faiss",
    metadata_path="vector_store/metadata.json"
)
generator = GeminiGenerator(model_name="gemini-2.5-flash")

# 3️⃣ Request/Response schemas
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

# 4️⃣ Endpoint
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
    ans = generator.generate(question=q, context=ctx, documents=docs)

    # record bot reply
    memory.add_bot_message(ans)

    return QueryResponse(answer=ans)
