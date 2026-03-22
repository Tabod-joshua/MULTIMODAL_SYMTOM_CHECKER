# `app.py` — Complete Line-by-Line Code Explanation

This is the **main brain of the DermLIP server**. It runs as a FastAPI web service inside a Docker container and handles everything from loading the AI model to returning a structured medical response. Every time a user sends a picture, this file processes it from start to finish.

---

## Part 1: Imports & System Path (Lines 1-15)

### Lines 1-2: System Path Injection
```python
import sys
sys.path.insert(0, "/app/Derm1M/src")
```
- **`import sys`**: Imports Python's built-in system utilities module.
- **`sys.path.insert(0, "/app/Derm1M/src")`**: This is a critical workaround. The DermLIP model was trained using a modified fork of `open_clip` that lives inside the `Derm1M` repository. Python's normal module search wouldn't find it. By inserting its path at position `0` (the front), Python will look there *first* before anywhere else. Without this, importing `open_clip` would load the wrong version.

### Lines 4-15: Standard and Third-Party Imports
```python
import re, uuid, json, torch, io, open_clip, httpx
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
```
- **`re`**: Regular expressions — used to detect LLM looping/stutter and strip markdown artifacts.
- **`uuid`**: For generating globally unique session IDs (e.g. `"bf0de27d-557f-40f4-8417-52dee42cd437"`). Every `/classify` call creates a new one.
- **`json`**: Parsing the `clinical_kb.json` database file.
- **`torch`**: PyTorch — the deep learning framework that powers the DermLIP CLIP model and runs it on GPU.
- **`io`**: In-memory byte stream handling — lets us process image bytes without saving anything to disk.
- **`open_clip`**: The modified CLIP implementation from Derm1M that can load the DermLIP pretrained weights.
- **`httpx`**: An async HTTP client used to call the Ollama LLM API. Chosen over `requests` because FastAPI is async and `requests` would block the server.
- **`from PIL import Image`**: Pillow — the standard Python imaging library for decoding JPEG/PNG/BMP to pixel arrays.
- **`from fastapi import FastAPI, File, UploadFile`**: The core web framework. `UploadFile` specifically handles HTTP multipart file uploads (when a user sends an image).
- **`from fastapi.responses import JSONResponse`**: Lets us return custom JSON with specific HTTP status codes.
- **`from huggingface_hub import hf_hub_download`**: Downloads the DermLIP checkpoint weights from HuggingFace Hub on first startup.
- **`from pydantic import BaseModel`**: Data validation for request bodies — used to define the `ChatRequest` schema.

---

## Part 2: App & Global Config (Lines 17-43)

### Line 17: App Instance
```python
app = FastAPI()
```
Creates the FastAPI application object. All `@app.get` / `@app.post` decorators below register routes to this single app instance.

### Lines 19-24: Session Store
```python
SESSIONS: dict[str, dict] = {}
```
An in-memory Python dictionary. When a user classifies an image, a session is created and stored here as `SESSIONS[uuid] = { condition, explanation, chat_history, ... }`. The key is the UUID session ID returned to the client. The upgrade note warns this must be replaced with a database (like PostgreSQL) before deploying to multiple users, since an in-memory dict is lost on restart and not shared between server instances.

### Lines 26-28: Device Selection
```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
```
At startup, checks if a CUDA-compatible NVIDIA GPU is available in the container. If yes, all PyTorch tensors (model inputs/outputs) are placed on the GPU. If not, falls back to CPU. The NVIDIA GPU provides ~50-100x faster inference for the CLIP model.

### Lines 30-33: Ollama API Config
```python
OLLAMA_GENERATE_URL = "http://host.docker.internal:11434/api/generate"
OLLAMA_CHAT_URL    = "http://host.docker.internal:11434/api/chat"
OLLAMA_MODEL = "gemma3n:latest"
```
- **`host.docker.internal`**: A special Docker DNS name that resolves to the *host machine's* IP from inside a container. This is how the Docker container talks to the Ollama app running natively on Windows.
- **Port 11434**: Ollama's default listening port.
- **`/api/generate`**: Used for the classify endpoint to rewrite the deterministic template (one-shot generation).
- **`/api/chat`**: Used for the follow-up conversation endpoint (maintains message history context).
- **`gemma3n:latest`**: The specific LLM model. Gemma 3N is Google's quantized medical LLM.

### Lines 35-43: Medical Disclaimer
```python
DISCLAIMER = ("⚠️ IMPORTANT MEDICAL DISCLAIMER: ...")
```
A hardcoded, legally mandatory string attached to **every single response**. It appears in both `/classify` and `/chat` JSON responses. The key reason: this is an AI tool, not a licensed medical device. The disclaimer protects against users relying solely on the AI for clinical decisions.

---

## Part 3: Model Loading (Lines 45-63)

### Lines 46-50: Download Model Weights
```python
checkpoint_path = hf_hub_download(
    repo_id="redlessone/DermLIP_ViT-B-16",
    filename="open_clip_pytorch_model.bin"
)
```
On first startup, this downloads the DermLIP model checkpoint (pre-trained weights) from HuggingFace Hub. The file is cached in `/root/.cache/huggingface` (which is a Docker volume so it persists across restarts). On subsequent starts, it loads from the local cache instantly.

### Lines 52-62: Initialize the Model
```python
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16")
open_clip.load_checkpoint(model, checkpoint_path)
tokenizer = open_clip.get_tokenizer("ViT-B-16")
model = model.to(DEVICE)
model.eval()
```
- **`create_model_and_transforms`**: Creates the Vision Transformer architecture (ViT-B/16) with its image preprocessing pipeline but without any weights loaded.
- **`load_checkpoint`**: Loads the DermLIP-specific weights into the blank architecture. This is done in two steps because the Derm1M factory has a missing import bug in the all-in-one `load_pretrained` path.
- **`get_tokenizer`**: Creates the text tokenizer that converts condition name strings (like "acne vulgaris...") into token IDs the model can process.
- **`.to(DEVICE)`**: Moves all model parameters to GPU memory.
- **`.eval()`**: Puts model in inference mode — disables dropout and batch normalization that are only relevant during training.

---

## Part 4: Conditions, KB & Hierarchy Loading (Lines 65-155)

### `_load_conditions()` — Lines 74-89
Reads `conditions.txt` line by line, skipping comment lines (`#`) and blank lines. The result is a Python list of the full condition descriptor strings (e.g. `"acne vulgaris — red inflamed papules..."`). This is the universe CLIP compares images against.

### `_load_clinical_kb()` — Lines 91-115
Reads `clinical_kb.json` and validates the schema of EVERY entry at startup, checking for required keys (`condition_id`, `display_name`, `description`, `emergency_symptoms`, etc.). **It throws a `RuntimeError` on startup if any key is missing** — fail loud, never silent.

### `_load_hierarchy_map()` — Lines 120-129 *(Phase 2 Addition)*
```python
HIERARCHY_MAP = _load_hierarchy_map("/app/hierarchy_map.json")
```
Loads the pre-built `hierarchy_map.json` (generated by `build_hierarchy.py`). This maps every condition key to one of **11 super-categories** (e.g. `"tinea_corporis" -> "Fungal Infections"`). If the file is missing, warns and falls back gracefully to flat search.

### `_build_category_buckets()` — Lines 131-150 *(Phase 2 Addition)*
At startup, builds a reverse lookup: **super-category → list of full condition strings**. This is the structure used during Pass 2 classification to restrict the CLIP search pool.

```python
CATEGORY_BUCKETS = { "Fungal Infections": ["tinea corporis...", ...], ... }
CATEGORY_NAMES   = ["Acne & Follicular", "Autoimmune & Inflammatory", ...]
```

### Module-Level Loading
```python
SKIN_CONDITIONS = _load_conditions(_CONDITIONS_FILE)  # 277 entries
CLINICAL_KB     = _load_clinical_kb("/app/clinical_kb.json")  # 278 entries (incl. fallback)
HIERARCHY_MAP   = _load_hierarchy_map("/app/hierarchy_map.json")  # 277 mappings
```
All loaded at import time — once globally. This is critical for performance.

---

## Part 5: CLIP Templates (Lines 156-172)

```python
TEMPLATES = [
    "a clinical photograph of {}",
    "a close-up photo of {}",
    "a dermatologist image showing {}",
    "a macro photo of {} on the skin",
    "a medical image of {}",
    "a dermoscopy image of {}",
    "a smartphone photo of {} on dark skin",
    "a smartphone photo of {} on light skin",
    "a well-lit clinical photo of {}",
    "a poor-quality image of {}",     # handles bad lighting
    "a detailed view of {}",
    "a pathology image indicating {}",
    "a zoomed-in photo of {}",
    "a picture of a skin lesion indicating {}",
    "{}"  # raw condition string as pure symptom descriptor
]
```
**15 templates** (Phase 2 expanded from 5). Each full condition string is embedded under all 15 frames. The per-template probabilities are mathematically averaged together — this is **prompt ensembling**, and measurably raises zero-shot accuracy by forcing the vision-language model to consider multiple real-world photo scenarios.


---

## Part 6: Helper Functions (Lines 132-181)

### `clean_display_name()` (Lines 133-148)
Converts a raw long condition string like `"acne vulgaris — common papulopustular acne with..."` into a clean 2-3 word display name like `"Acne Vulgaris"`. It strips em-dashes and hyphens first, then takes the first 2-3 non-generic words and applies `.title()` casing. This name appears in the UI and is also used as the diagnosis keyword for LLM validation.

### `build_raw_template()` (Lines 153-167)
Takes a `kb_entry` dictionary from `clinical_kb.json` and formats it into a structured text block:
```
Condition: [Name]
Description: [text]
Common causes: [text]
General management: [text]
Seek medical care if: [text]
```
This is the **deterministic output** — 100% controlled, no LLM involved. It's what gets sent to the LLM for polishing, and it's what gets returned in `raw_explanation` in the API.

### `build_polish_prompt()` (Lines 169-181)
Wraps the raw template in Gemma's special instruct formatting tags (`<start_of_turn>user`, `<start_of_turn>model`). These tags tell Gemma it is in an instruction-following dialogue. The prompt explicitly instructs: keep the condition name exactly, maintain all medical facts, don't add new information, return plain text only. This is the patient-friendly language safety constraint.

---

## Part 7: `/health` Endpoint (Lines 185-187)
```python
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "llm": OLLAMA_MODEL}
```
A simple status check used by `evaluate.py` at the start of every test run. Returns the current GPU/CPU device and LLM model name. Useful for deployment monitoring.

---

## Part 8: `/classify` Endpoint (Lines 190-382)

This is the core of the entire system. It runs a 5-layer pipeline:

### Layer 1: Test-Time Augmentation — TTA (Phase 2)
```python
aug_images = [
    base_image,                                    # Original
    base_image.transpose(FLIP_LEFT_RIGHT),         # Flipped
    ImageEnhance.Brightness(base_image).enhance(1.15), # Brighter
    ImageEnhance.Contrast(base_image).enhance(1.15),   # High contrast
    base_image.rotate(15, resample=BICUBIC)        # Rotated 15°
]
image_tensors = torch.cat([preprocess(img).unsqueeze(0) for img in aug_images]).to(DEVICE)
```
Instead of processing the raw photo once, 5 augmented variants are spawned in memory. This makes the system resilient to poor camera angles, dark lighting, off-center lesions, or flash glare. All 5 are processed in a **single batched GPU call** → shape `[5, 3, 224, 224]`.

### Layer 2: Hierarchical 2-Pass Classification (Phase 2)
```
Pass 1 (Coarse) → Lock super-category from 11 broad labels
Pass 2 (Fine)   → Search only within that category's conditions
```

**Pass 1 — Super-Category Locking:**
```python
# TTA images → encode → average into one representative vector
avg_image_features = image_features.mean(dim=0, keepdim=True)
# Compare against 11 broad labels
cat_logits = (100.0 * avg_image_features @ cat_features.T).softmax(dim=-1)
locked_category = max(category_scores, key=category_scores.get)
```
The 5 image variants are averaged into one representative vector, then compared against 11 broad category labels (e.g. `"skin condition related to fungal infections"`). The winner is **locked** — the model is now committed to a specific disease family. Printed to logs for transparency.

**Pass 2 — Fine-Grained Search Within Bucket:**
```python
search_pool = CATEGORY_BUCKETS[locked_category]  # e.g. 12 fungal conditions
# Full 15-template ensembling on all 5 TTA images vs. search_pool only
logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
```
Full 15-template prompt ensembling is run, but **only against the conditions in the locked bucket** instead of all 277. This is the crucial accuracy gain — a fungal infection can now only be misclassified as *another fungal infection*, never as a melanoma or autoimmune disease.

**Fallback safety**: If for any reason the hierarchy map fails or returns an empty bucket, the code falls back to the full 277-condition vocab automatically.

### Layer 3: KB Lookup (unchanged)
The top-ranked condition string is normalized and looked up in `CLINICAL_KB`. Falls back to `generic_fallback`.

### Layer 4: Optional LLM Polish (unchanged)
Deterministic template → Gemma restyles to be patient-friendly. Rejected if diagnosis keywords drift. Bypassed if `skip_llm=True`.

### Layer 5: Response
Now includes two new fields:
```json
"hierarchical_category": "Fungal Infections",
"category_confidence": "71.2%"
```
This gives full transparency about the hierarchical decision for debugging and clinical context.

---

## Part 9: `/chat` Endpoint (Lines 384-485)

### ChatRequest Model (Lines 386-388)
```python
class ChatRequest(BaseModel):
    session_id: str
    message: str
```
Pydantic validates that every incoming POST body has exactly these two fields with these types.

### Conversation Logic (Lines 392-479)
1. **Session lookup**: Retrieves the session by ID. Returns a 404 if not found (user hasn't classified an image yet).
2. **Self-care directive**: If the KB says this condition has `allow_self_care: true`, the system prompt allows general skincare advice. Otherwise, it instructs the LLM to defer to a doctor.
3. **Message history**: The last 6 conversation turns are injected as alternating `user`/`assistant` messages, giving the LLM conversational context memory.
4. **LLM call**: Uses `/api/chat` (the stateful chat format) rather than `/api/generate` (one-shot format).
5. **Loop detection**: The `_is_looping()` function checks for token repetition, vocabulary collapse, and sentence-level duplicates.
6. **History save**: The user's question and the assistant's reply are appended to `chat_history` for the next turn.
