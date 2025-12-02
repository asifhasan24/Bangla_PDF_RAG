
from celery_app import celery_app
from generator import GeminiGenerator
from preprocess import extract_text
from chunker import sentence_chunks
from embedder import Embedder
import os
import json
from config import VECTOR_STORE_DIR

generator = GeminiGenerator(model_name="gemini-2.5-flash")
embedder = Embedder(model_name="sentence-transformers/LaBSE") # Assuming this is the model used for embedding

@celery_app.task
def generate_answer_task(question, context, documents):
    """
    Celery task to generate an answer.
    """
    ans = generator.generate(question=question, context=context, documents=documents)
    return ans

@celery_app.task
def process_document_task(file_path: str):
    """
    Celery task to process an uploaded document, chunk it, embed it, and update the vector store.
    """
    try:
        # 1. Extract text
        text = extract_text(file_path)

        # 2. Chunk text
        chunks = [
            {"id": idx, "text": chunk}
            for idx, chunk in enumerate(sentence_chunks(text, max_sentences=5))
        ]

        # 3. Embed and build index
        # Define paths for the new index and metadata
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        index_path = os.path.join(VECTOR_STORE_DIR, f"{base_filename}.faiss")
        metadata_path = os.path.join(VECTOR_STORE_DIR, f"{base_filename}.json")
        
        embedder.build_index(chunks, index_path, metadata_path)

        # Clean up the uploaded file
        os.remove(file_path)

        return {"status": "success", "message": f"Document processed and vector store updated from {file_path}"}
    except Exception as e:
        # Clean up the uploaded file even if processing fails
        if os.path.exists(file_path):
            os.remove(file_path)
        return {"status": "error", "message": str(e)}
