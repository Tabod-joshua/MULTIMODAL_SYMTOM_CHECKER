# `docker-compose.yml` — Complete Line-by-Line Code Explanation

`docker-compose.yml` is the **orchestration file** that tells Docker how to *run* and *wire together* the container(s) built by the Dockerfile. While the `Dockerfile` is about *building* the image, `docker-compose.yml` is about *launching* it with the right ports, volumes, and hardware access.

---

## Line 1: Services Block
```yaml
services:
```
The top-level `services` key. Everything under this is a named service (a container to start). We only have one: `dermlip`.

---

## Line 2-3: Service Name & Build Instructions
```yaml
  dermlip:
    build: .
```
- **`dermlip`**: The arbitrary name of our service. Referenced in commands like `docker compose logs dermlip-1`.
- **`build: .`**: Tells compose to build the image using the `Dockerfile` in the current directory (`.`). This is why you run `docker compose up --build` after changing the Dockerfile.

---

## Lines 4-5: Port Mapping
```yaml
    ports:
      - "8000:8000"
```
- **`"8000:8000"`**: Maps port 8000 on your Windows host machine to port 8000 inside the container.
- **Format**: `"HOST_PORT:CONTAINER_PORT"`
- **Why we need it**: The container's network is isolated by default. Without this, `http://localhost:8000` from Windows would not reach the FastAPI server inside the container. This mapping punches a hole in the isolation for exactly this one port.

---

## Lines 6-11: Volume Mounts
```yaml
    volumes:
      - hf_cache:/root/.cache/huggingface
      - ./conditions.txt:/app/conditions.txt:ro
      - ./clinical_kb.json:/app/clinical_kb.json:ro
      - ./hierarchy_map.json:/app/hierarchy_map.json:ro
      - ./app.py:/app/app.py:ro
```
Volumes tell Docker how to share files between your local machine and the container.

- **`hf_cache:/root/.cache/huggingface`**: A named volume. On first startup, `hf_hub_download()` downloads the ~400MB DermLIP model weights and caches them here. Persists automatically between restarts — no re-download.
- **`./conditions.txt:/app/conditions.txt:ro`**: Binds `conditions.txt` from your local disk into the container as Read-Only. Edit → restart → server picks it up instantly. **No rebuild required.**
- **`./clinical_kb.json:/app/clinical_kb.json:ro`**: Same pattern for the clinical knowledge base. All 278 condition schemas available on restart.
- **`./hierarchy_map.json:/app/hierarchy_map.json:ro`** *(Phase 2 Addition)*: The super-category bucketing file used for Hierarchical Classification. Maps all 277 conditions to one of 11 broad disease classes. Regenerate with `python build_hierarchy.py` if you add new conditions, then restart.
- **`./app.py:/app/app.py:ro`**: The server code itself. Combined with `--reload`, editing `app.py` causes uvicorn to hot-reload without a rebuild.

> **When do you need a full rebuild (`docker compose up --build`)?**
> Only if you change the `Dockerfile` itself or add/remove Python packages in `requirements.txt`.
> Changing `app.py`, `conditions.txt`, `clinical_kb.json`, or `hierarchy_map.json` only requires a **restart** (`docker compose down && docker compose up`).


---

## Lines 11-18: GPU Resource Reservation
```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [ gpu ]
```
- **`deploy.resources.reservations`**: Tells Docker Compose to reserve hardware resources for this service.
- **`driver: nvidia`**: Specifies we want an NVIDIA GPU (requires the NVIDIA Container Toolkit to be installed on Windows).
- **`count: 1`**: Reserve exactly 1 GPU.
- **`capabilities: [ gpu ]`**: Exposes the GPU compute capability to the container.
- **Why we need it**: Without this block, Docker gives the container NO GPU access. PyTorch would fall back to CPU, making the CLIP inference ~50-100x slower and making the DermLIP model unusably slow.

---

## Line 19: Restart Policy
```yaml
    restart: "no"
```
- **What it does**: If the container crashes, Docker will NOT automatically restart it.
- **Why `"no"` instead of `"always"`**: During development and testing, we want container crashes to be visible and explicit. If it automatically restarted, a boot-time error (like a bad `conditions.txt`) could loop silently and be hard to diagnose. For production deployment, this would be changed to `restart: unless-stopped`.

---

## Lines 21-22: Named Volume Definition
```yaml
volumes:
  hf_cache:
```
- **What it does**: Declares `hf_cache` as a named volume managed by Docker.
- **Why we need it**: Named volumes must be declared here at the top-level to be used in the `services` block above. Docker creates and manages the storage location automatically (typically under `C:\ProgramData\docker\volumes\` on Windows).
