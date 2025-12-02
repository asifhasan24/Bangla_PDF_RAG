import os

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Vector store directory
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")

# Data directory
DATA_DIR = os.path.join(BASE_DIR, "data")

# Upload directory
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

# Ensure directories exist
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Default index and metadata paths
DEFAULT_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "index.faiss")
DEFAULT_METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "metadata.json")
