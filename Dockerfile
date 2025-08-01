# Use official slim Python image
FROM python:3.10-slim

# Ensure stdout/stderr are unbuffered (optional, helps in logs)
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && python - <<EOF
import nltk
nltk.download('punkt_tab')
EOF

# Copy application code
COPY . .

# Expose the API port
EXPOSE 8000

# Default command: start FastAPI server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
