# Hugging Face Spaces Docker SDK Image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOST=0.0.0.0 \
    DATA_DIR=/app/knowledge_base \
    INDEX_DIR=/app/knowledge_base/indexes \
    PROCESSED_DIR=/app/knowledge_base/processed \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-cache embedding model weights into Docker image at build time
RUN python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small'); AutoModel.from_pretrained('intfloat/multilingual-e5-small')"

# Copy application source code and pre-built index/processed artifacts
COPY config.py .
COPY app.py .
COPY knowledge_base/ ./knowledge_base/
COPY doc_chunking/ ./doc_chunking/
COPY vector_search/ ./vector_search/
COPY voice_stt/ ./voice_stt/
COPY safety_guardrails/ ./safety_guardrails/
COPY llm_synthesis/ ./llm_synthesis/
COPY rag_pipeline/ ./rag_pipeline/
COPY api_endpoints/ ./api_endpoints/
COPY latency_benchmarks/ ./latency_benchmarks/
COPY web_ui/ ./web_ui/

# Create user for Hugging Face Spaces security
RUN useradd -m -u 1000 user && \
    chown -R user:user /app
USER user

# Expose default Hugging Face Spaces port
EXPOSE 7860

# Launch FastAPI server
CMD ["python", "app.py"]
