FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Enable BuildKit for cache mounts
WORKDIR /app

# Configure pip to use cache directory
ENV PIP_CACHE_DIR=/root/.cache/pip

RUN apt-get update -o APT::Acquire::max_concurrent_connections=5 \
    -o APT::Acquire::retry=3 \
    -o APT::Acquire::http::timeout=30 \
    && apt-get install -y \
    python3 python3-pip git \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Clone the Derm1M repo (contains the modified open_clip needed)
RUN git clone https://github.com/SiyuanYan1/Derm1M.git /app/Derm1M

# Patch Derm1M's factory.py: add missing get_pretrained_cfg import
RUN sed -i 's/from .pretrained import download_pretrained,\\/from .pretrained import download_pretrained, get_pretrained_cfg,\\/' /app/Derm1M/src/open_clip/factory.py

# Install Derm1M dependencies EXCLUDING torch/torchvision (will install last)
WORKDIR /app/Derm1M
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -v "^torch" requirements.txt | grep -v "^torchvision" | sed '/--hash/d;s/#sha256.*//' > /tmp/derm1m_filtered.txt && \
    pip3 install --default-timeout=1000 --retries 5 -r /tmp/derm1m_filtered.txt

# Install FastAPI server deps
RUN --mount=type=cache,target=/root/.cache/pip pip3 install \
    --default-timeout=1000 --retries 5 \
    fastapi uvicorn python-multipart huggingface_hub

# Install PyTorch with CUDA 12.8 LAST — do NOT pin torchvision separately,
# let pip resolve the correct version from the cu128 index
RUN --mount=type=cache,target=/root/.cache/pip pip3 install \
    --default-timeout=1000 --retries 5 \
    torch==2.7.0+cu128 torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# Copy our API server
COPY app.py /app/app.py

WORKDIR /app
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]