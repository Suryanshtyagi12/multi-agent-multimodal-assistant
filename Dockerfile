FROM python:3.11-slim

# System dependencies for Docling layout detection
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install CPU-only PyTorch first to avoid CUDA downloads
# This saves ~2GB on the Docker image size
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create required directories
RUN mkdir -p figures chroma_db

# Fix Docling symlink issue on Linux
ENV DOCLING_ARTIFACTS_PATH=/app/.docling_cache
RUN mkdir -p /app/.docling_cache

# Expose both ports
# 8000 = FastAPI backend
# 7860 = Streamlit frontend (required by HuggingFace Spaces)
EXPOSE 8000 7860

# Copy and set permissions for startup script
COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]
