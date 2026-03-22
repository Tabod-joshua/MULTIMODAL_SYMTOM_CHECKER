# Copilot Instructions - DermLIP Server

## Project Overview
**DermLIP** is a medical-grade dermatology diagnostic API that classifies skin conditions from images and provides patient-friendly explanations via LLM integration. It's containerized with CUDA support for GPU acceleration.

### Architecture
- **API Server**: FastAPI/Uvicorn on port 8000
- **Vision Model**: DermLIP (CLIP-based, ~150MB weights from HuggingFace)
- **LLM Integration**: Ollama `medgemma` running on `host.docker.internal:11434` (must be pre-running)
- **Execution**: Docker container with CUDA 12.8 (compatible with host CUDA 13.1+)

### Data Flow
1. User uploads skin image → `POST /classify` endpoint
2. Image preprocessed and encoded via DermLIP vision encoder
3. Compared against 200+ curated skin condition embeddings (with template: "a clinical photograph of {condition}")
4. Top 4 predictions ranked by cosine similarity
5. Top prediction + alternatives sent to Ollama/MedGemma with structured prompt
6. LLM returns plain-language explanation with home care & emergency warning signs
7. Response includes: medical disclaimer, classification scores, LLM explanation, emergency criteria

## Critical Developer Context

### Medical & Safety Requirements
- **ALL responses must include the DISCLAIMER** (non-negotiable, hardcoded in every classification response)
- Emphasis: AI results are NOT diagnoses, patients MUST see doctors
- Target audience: Tropical/West African regions (references Cameroon epidemiology)
- `seek_emergency_care_immediately_if` list is critical safety feature

### External Dependencies (Must Be Running)
| Service | Host | Port | Purpose | Failure Handling |
|---------|------|------|---------|------------------|
| Ollama | `host.docker.internal:11434` | 11434 | LLM explanations | Graceful error message, no crash |
| HuggingFace Hub | Auto-download at startup | — | DermLIP weights (~150MB) | Blocks container startup |

### Skin Conditions Data Structure
- **200+ conditions** curated by epidemiological region (Cameroon prevalence percentages noted in comments)
- Fungal infections (25.4%), Parasitic (21.4%), Inflammatory (34.3%), Acne (14.6%) are highest prevalence
- Format: Single multi-word string per condition (`"tinea capitis scalp ringworm fungal infection"`)
- Used with template prefix: `"a clinical photograph of " + condition`
- **Never truncate this list** - comprehensiveness is medical accuracy requirement

### Build & Runtime
- **Dockerfile workflow**: 
  1. Base: `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04`
  2. Clone `SiyuanYan1/Derm1M` repo (contains modified `open_clip`)
  3. Install Derm1M's requirements.txt
  4. Install PyTorch 2.3.0+cu128 with CUDA extras
  5. Install FastAPI deps
  6. Copy app.py
- **GPU Detection**: Device auto-switches to CPU if no CUDA available (`DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`)
- **Container isolation**: CUDA 12.8 in container is separate from host (doesn't override host CUDA 13.1)

## Key Code Patterns

### Image Classification Pipeline
```python
# Preprocess image (handles different formats via PIL)
image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

# Batch encode all conditions (200+ text embeddings)
text_inputs = tokenizer([TEMPLATE + c for c in SKIN_CONDITIONS]).to(DEVICE)

# CLIP similarity: normalized dot product → softmax
logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
```
- **Note**: Softmax scaling by 100 increases confidence separation (CLIP best practice)
- Output is probability distribution across all 200+ conditions

### LLM Integration Pattern
```python
# Build dynamic prompt with top prediction + alternatives
prompt = build_prompt(top_condition, top_confidence, alternatives)

# Async call to Ollama with structured response format
async with httpx.AsyncClient(timeout=90.0) as client:
    response = await client.post(OLLAMA_URL, json={...})
```
- **Timeout**: 90s for LLM generation (medical explanations are lengthy)
- **Error handling**: If Ollama fails, returns graceful error message (doesn't crash classification)
- **Temperature**: 0.3 (low) for factual, consistent medical explanations
- **Max tokens**: 700 (constrains explanation length to ~400 words as per prompt)

## Modification Guidelines

### When Adding/Modifying Skin Conditions
1. Add to `SKIN_CONDITIONS` list with full descriptive string
2. Include common names + medical terms + symptoms (improves CLIP matching)
3. Consider epidemiological context (note regional prevalence if regional-specific)
4. Test via `/classify` endpoint with representative images

### When Updating LLM Prompts
- The 5-section format (`WHAT IS THIS?`, `SIGNS TO WATCH FOR`, `WHAT YOU CAN DO`, `WHEN TO SEE A DOCTOR`, `IMPORTANT REMINDER`) is deliberate and tested with MedGemma
- Keep language simple (no medical jargon, tropical context)
- Ensure final reminder about non-diagnosis is prominent
- Test temperature/num_predict values with real Ollama instance (affects output quality)

### When Modifying Docker Build
- PyTorch version must match CUDA 12.8 (specified as `torch==2.3.0+cu128`)
- If updating base image, verify CUDA version compatibility with PyTorch version
- Derm1M repo is cloned fresh each build (contains modified `open_clip` - don't use PyPI version)

## Local Development Workflow

1. **Ensure Ollama is running**: `ollama serve` in separate terminal on host
2. **Pull base image**: `docker pull nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04`
3. **Build & run**: `docker compose up -d --build`
4. **Test health**: `curl http://localhost:8000/health`
5. **Test classification**: `curl -X POST -F "file=@image.jpg" http://localhost:8000/classify`

### Debugging
- Docker logs: `docker logs dermlip-server-dermlip-1`
- GPU access: Check `docker exec {container} nvidia-smi`
- Model loading: Check stderr for HuggingFace download progress
- Ollama timeout: If 90s → increase timeout, check Ollama is responsive

## Testing Recommendations
- Unit: Mock image preprocessing, mock text encoding, verify confidence ranking
- Integration: Real image + real Ollama instance, verify full response structure
- Medical: Validate response includes disclaimer, emergency signs, top alternatives
- Performance: Monitor GPU memory (DermLIP + PyTorch are memory-intensive), profile text encoding batch

## Common Issues & Solutions

### Docker Build Hangs on PyTorch Install
- PyTorch with CUDA is large (~2-3GB). 5-30min is normal.
- Check disk space: `docker system df`
- If persists: `docker builder prune -af` then rebuild

### Ollama Connection Refused
- Verify Ollama running: `curl http://host.docker.internal:11434/api/tags`
- Container can't resolve `host.docker.internal` on Linux? Use `host=0.0.0.0` and network mode
- Increase timeout if Ollama is slow: `timeout=120.0`

### Model Weights Download Fails
- Check HuggingFace connectivity and repo exists: `redlessone/DermLIP_ViT-B-16`
- May need HF token if private: set `HF_TOKEN` env var in Dockerfile
- Weights cached after first download

### Low GPU Memory
- Reduce batch size (currently 1)
- Disable gradients: Already using `torch.no_grad()` ✓
- Offload to CPU if needed (see DEVICE logic)
