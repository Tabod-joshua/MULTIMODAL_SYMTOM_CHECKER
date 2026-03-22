# `Dockerfile` — Complete Line-by-Line Code Explanation

The Dockerfile is the **recipe for building the DermLIP Docker container**. It describes, step by step, how to go from a blank Linux machine with an NVIDIA GPU driver to a fully running DermLIP API server. Docker executes these instructions top-to-bottom and caches each layer so unchanged steps don't re-run.

---

## Line 1: Base Image
```dockerfile
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04
```
- **What it does**: Sets the starting point of the container. Instead of starting from scratch, we start from an official NVIDIA image that already has:
  - Ubuntu 22.04 (the Linux OS)
  - CUDA 12.8 runtime libraries (the GPU communication layer)
  - cuDNN (CUDA Deep Neural Network library — optimized math ops for neural networks)
- **Why we need it**: PyTorch requires CUDA libraries to use the GPU. Installing those from scratch is extremely complex. Starting from an NVIDIA base image means GPU support is pre-configured and guaranteed compatible.
- **`runtime` vs `devel`**: We use the `runtime` image, not `devel`. The `devel` image includes compilers needed to BUILD CUDA code. We only need the runtime (to RUN code), which is a smaller image.

---

## Line 3: BuildKit Comment
```dockerfile
# Enable BuildKit for cache mounts
```
A comment. It notes that `--mount=type=cache` (used later) requires Docker BuildKit, which is enabled by default in modern Docker versions.

---

## Line 4: Working Directory
```dockerfile
WORKDIR /app
```
- **What it does**: Sets `/app` as the current working directory for all subsequent commands. If the directory doesn't exist, Docker creates it automatically.
- **Why we need it**: Organizes files logically. Instead of files scattering across the filesystem root, everything lives under `/app`.

---

## Line 7: pip Cache Directory
```dockerfile
ENV PIP_CACHE_DIR=/root/.cache/pip
```
- **What it does**: Sets an environment variable that tells pip where to store its download cache.
- **Why we need it**: Lines 25-27, 30-32, 36-39 use `--mount=type=cache,target=/root/.cache/pip`. This mounts a persistent cache volume at that path during build, so pip downloads are not repeated if the Dockerfile is rebuilt. Without this, every `RUN pip install` would re-download packages from the internet even if nothing changed.

---

## Lines 9-15: System Dependencies
```dockerfile
RUN apt-get update ... && apt-get install -y \
    python3 python3-pip git \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*
```
- **`apt-get update`**: Updates the package repository index.
  - `-o APT::Acquire::max_concurrent_connections=5`: Downloads up to 5 packages in parallel for speed.
  - `-o APT::Acquire::retry=3`: Retries failed downloads up to 3 times.
  - `-o APT::Acquire::http::timeout=30`: Waits max 30 seconds per package.
- **`python3 python3-pip`**: Installs Python 3 and its package manager.
- **`git`**: Required by the next step to clone the Derm1M repository.
- **`libglib2.0-0 libsm6 libxrender1 libxext6`**: C-level system libraries required by OpenCV and Pillow when processing images. Without these, `import cv2` or `PIL.Image.open()` would crash with missing shared library errors.
- **`rm -rf /var/lib/apt/lists/*`**: Deletes the package index cache after installation to reduce the image size by ~50MB.

---

## Line 18: Clone Derm1M Repository
```dockerfile
RUN git clone https://github.com/SiyuanYan1/Derm1M.git /app/Derm1M
```
- **What it does**: Downloads the entire Derm1M research repository to `/app/Derm1M`.
- **Why we need it**: Derm1M contains a *modified* fork of `open_clip` with DermLIP-specific changes. The standard PyPI `open-clip-torch` package doesn't include the architectural modification needed to load the DermLIP weights. Without this custom fork, the model cannot be loaded at all.

---

## Line 21: Patch Factory Bug
```dockerfile
RUN sed -i 's/from .pretrained import download_pretrained,\\/from .pretrained import download_pretrained, get_pretrained_cfg,\\/' /app/Derm1M/src/open_clip/factory.py
```
- **What it does**: Uses the `sed` stream editor to insert `get_pretrained_cfg,` into an import line in `factory.py`.
- **Why we need it**: The Derm1M repository has a bug in its `factory.py` file — a function called `get_pretrained_cfg` is used but never imported. When left unfixed, the server crashes with `ImportError: cannot import name 'get_pretrained_cfg'` during model loading. Since modifying the source file directly and committing it would couple our project to the upstream repo, patching it as a Docker build step is the cleaner, more maintainable approach.

---

## Lines 24-27: Derm1M Dependencies (Filtered)
```dockerfile
WORKDIR /app/Derm1M
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -v "^torch" requirements.txt | grep -v "^torchvision" | sed '/--hash/d;s/#sha256.*//' > /tmp/derm1m_filtered.txt && \
    pip3 install --default-timeout=1000 --retries 5 -r /tmp/derm1m_filtered.txt
```
- **`WORKDIR /app/Derm1M`**: Changes directory so the next command can reference the Derm1M `requirements.txt`.
- **`grep -v "^torch"`**: Removes any line starting with `torch` from the requirements file — we don't want Derm1M's pinned PyTorch version.
- **`grep -v "^torchvision"`**: Similarly removes `torchvision`.
- **`sed '/--hash/d;s/#sha256.*//'`**: Strips hash integrity checks (`--hash=sha256:...`) that would conflict with our custom index.
- **Output to `/tmp/derm1m_filtered.txt`**: Saves the cleaned requirements to a temp file.
- **`pip3 install ... -r /tmp/derm1m_filtered.txt`**: Installs all Derm1M dependencies *except* PyTorch (which we install separately with the correct CUDA version). `--default-timeout=1000` and `--retries 5` handle slow or flaky network connections.
- **Why separate?**: If we let Derm1M bring in its own `torch` version, it would likely install a CPU-only version or the wrong CUDA version, breaking GPU support.

---

## Lines 29-32: FastAPI Server Dependencies
```dockerfile
RUN --mount=type=cache,target=/root/.cache/pip pip3 install \
    --default-timeout=1000 --retries 5 \
    fastapi uvicorn python-multipart huggingface_hub
```
- **`fastapi`**: The web framework for our `/classify` and `/chat` API endpoints.
- **`uvicorn`**: The ASGI server that actually runs FastAPI (like Gunicorn but async).
- **`python-multipart`**: Enables FastAPI to parse `multipart/form-data` HTTP requests, which is required to accept uploaded image files.
- **`huggingface_hub`**: The client library for downloading model weights from HuggingFace Hub.
- **Why not in `requirements.txt`?**: The `requirements.txt` file exists but says "PyTorch is installed separately in Dockerfile." This split mirrors the Dockerfile's deliberate PyTorch-last ordering.

---

## Lines 34-39: PyTorch with CUDA 12.8
```dockerfile
RUN --mount=type=cache,target=/root/.cache/pip pip3 install \
    --default-timeout=1000 --retries 5 \
    torch==2.7.0+cu128 torchvision \
    --index-url https://download.pytorch.org/whl/cu128
```
- **`torch==2.7.0+cu128`**: PyTorch version 2.7.0 compiled for CUDA 12.8. The `+cu128` suffix is key — this is the GPU-enabled build.
- **`torchvision`**: Companion package (image transforms, pretrained vision models). We let pip resolve a compatible version rather than pinning it separately.
- **`--index-url https://download.pytorch.org/whl/cu128`**: PyTorch GPU builds are not on PyPI. They are hosted on PyTorch's custom wheel index, and this tells pip where to find them.
- **Why last?**: PyTorch is installed last because it's the heaviest package (~4GB) and has strict compatibility requirements with CUDA. Installing it after everything else ensures nothing later can downgrade or override it.

---

## Line 42: Copy Application Code
```dockerfile
COPY app.py /app/app.py
```
- **What it does**: Copies our `app.py` from the build context (the `dermlip-server` directory on your local machine) into the container at `/app/app.py`.
- **Why it's at the end**: Docker layers are cached. If `app.py` changes, only this layer and everything after it are re-run. All the expensive dependency-installation layers above are cached and reused. If `app.py` were near the top, every code change would trigger a full re-install of PyTorch.
- **Note**: In `docker-compose.yml`, `app.py` is also mounted as a volume override (`./app.py:/app/app.py:ro`), which means the live file on your disk replaces this baked-in copy at runtime without rebuilding.

---

## Lines 44-46: Final Configuration
```dockerfile
WORKDIR /app
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```
- **`WORKDIR /app`**: Resets the working directory back to `/app` after the Derm1M build steps.
- **`EXPOSE 8000`**: Documents that the container listens on port 8000. Does not actually publish the port — that's done by `docker-compose.yml`'s `ports:` section.
- **`CMD [...]`**: The default command that runs when the container starts.
  - **`uvicorn app:app`**: Run the `app` variable (the FastAPI instance) from the `app.py` file.
  - **`--host 0.0.0.0`**: Listen on all network interfaces (not just `localhost`) so Docker can route requests from outside the container.
  - **`--port 8000`**: The port to listen on.
  - **`--reload`**: Enables auto-reload when `app.py` changes on disk. Combined with the volume mount in `docker-compose.yml`, this means you can edit `app.py` and the server restarts automatically within a few seconds — no `docker compose restart` needed.
