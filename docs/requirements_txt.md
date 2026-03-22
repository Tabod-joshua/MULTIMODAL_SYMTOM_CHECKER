# `requirements.txt` — Complete Line-by-Line Code Explanation

`requirements.txt` defines the Python package dependencies for the DermLIP server. This file is referenced in the `Dockerfile` during the build process.

**Important architectural note:** This file is only a *partial* dependency list. PyTorch is intentionally excluded because installing it via this file would use the default PyPI index, which only has the CPU version of PyTorch. PyTorch with CUDA support must be installed separately with `--index-url https://download.pytorch.org/whl/cu128`, which is done directly in the Dockerfile.

---

```
# Web server
fastapi==0.111.0
uvicorn==0.30.1
python-multipart==0.0.9
```

### Line 2: `fastapi==0.111.0`
- **What it is**: The web framework powering all HTTP endpoints (`/health`, `/classify`, `/chat`).
- **Why this version**: Pinned to `0.111.0` for reproducibility. FastAPI changes rapidly, and a version bump could break the async behavior or the `UploadFile` API. Using a pinned version guarantees the container builds identically every time, on any machine.
- **What breaks without it**: Nothing would work. The entire `app.py` file relies on FastAPI's `@app.post()`, `UploadFile`, `JSONResponse`, and `File()` components.

### Line 3: `uvicorn==0.30.1`
- **What it is**: The ASGI (Asynchronous Server Gateway Interface) server that *runs* the FastAPI application.
- **Why we need it**: FastAPI defines the routes and logic, but something needs to actually listen on port 8000 and dispatch HTTP requests to FastAPI. Uvicorn does that. It's the difference between a Python script and a production-grade web server.
- **`uvicorn` vs `gunicorn`**: Uvicorn is chosen because FastAPI is async (`async def classify(...)`). Traditional WSGI servers like Gunicorn can't handle async natively, but Uvicorn is built for it.

### Line 4: `python-multipart==0.0.9`
- **What it is**: A parser for `multipart/form-data` HTTP requests.
- **Why we need it**: When the mobile app or `evaluate.py` uploads an image using `files={"file": fh}`, the HTTP request is encoded as `multipart/form-data`. This is the binary format used for all HTTP file uploads. **Without this package, FastAPI cannot parse file uploads at all** — it raises an error saying `Please install python-multipart`.
- **What breaks without it**: The `/classify` endpoint cannot receive image files.

---

```
# Model utilities
huggingface_hub==0.23.4
Pillow==11.0.0
numpy==2.1.0
```

### Line 7: `huggingface_hub==0.23.4`
- **What it is**: The official HuggingFace client library.
- **Why we need it**: `app.py` calls `hf_hub_download(repo_id="redlessone/DermLIP_ViT-B-16", filename="open_clip_pytorch_model.bin")` to download the DermLIP model weights at startup. This library handles authentication, caching, and downloading from HuggingFace's model repository servers.
- **What breaks without it**: The `import hf_hub_download` fails at startup and the model can never be loaded.

### Line 8: `Pillow==11.0.0`
- **What it is**: The Python Imaging Library (PIL) — the standard library for all image processing in Python.
- **Why we need it**: `app.py` uses `Image.open(io.BytesIO(image_bytes)).convert("RGB")` to decode the incoming uploaded image bytes into a pixel array that PyTorch can process. Pillow handles JPEG, PNG, BMP, RGBA, and all common image formats correctly.
- **What breaks without it**: `from PIL import Image` fails. The server cannot process any image.

### Line 9: `numpy==2.1.0`
- **What it is**: The foundational numerical computing library for Python. Provides N-dimensional array objects.
- **Why we need it**: While not explicitly `import numpy` in `app.py`, both Pillow and PyTorch depend on numpy's C-level array format for data interchange. Many image processing operations work through numpy arrays internally.
- **Why pinned**: numpy 2.x introduced breaking API changes vs numpy 1.x. Pinning to `2.1.0` ensures compatibility with both Pillow 11 and PyTorch 2.7.

---

```
# PyTorch is installed separately in Dockerfile with CUDA optimizations
# Do not add torch/torchvision here
```

### Lines 11-12: Comment
- **What it says**: An explicit architectural note warning future developers NOT to add `torch` to this file.
- **Why it matters**: If someone ran `pip install -r requirements.txt` locally (not through Docker), they would get a CPU-only torch from PyPI. The GPU-enabled version must be installed via `--index-url https://download.pytorch.org/whl/cu128`. This comment prevents a common mistake.
