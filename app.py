import sys
sys.path.insert(0, "/app/Derm1M/src")

import re
import uuid
import json
import torch
import io
import open_clip
import httpx
from PIL import Image, ImageEnhance
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download
from pydantic import BaseModel

app = FastAPI()

# ── Session Store ───────────────────────────────────────────────────────────────
# ⚠️  UPGRADE PATH: Replace this dict with a DB-backed session store
#    (e.g. SQLAlchemy + PostgreSQL) when adding user accounts / history.
#    The /classify and /chat endpoints only call get/set on SESSIONS, so
#    swapping to a DB is a small isolated change.
SESSIONS: dict[str, dict] = {}

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ── Ollama config ─────────────────────────────────────────────────────────────
OLLAMA_GENERATE_URL = "http://host.docker.internal:11434/api/generate"
OLLAMA_CHAT_URL    = "http://host.docker.internal:11434/api/chat"
OLLAMA_MODEL = "gemma3n:latest"

# ── Disclaimer (shown on every single response, no exceptions) ────────────────
DISCLAIMER = (
    "⚠️ IMPORTANT MEDICAL DISCLAIMER: THIS AI TOOL IS NOT A DOCTOR AND DOES NOT PROVIDE "
    "MEDICAL ADVICE, DIAGNOSIS, OR TREATMENT. THE RESULTS SHOWN ARE FOR INFORMATIONAL "
    "PURPOSES ONLY AND MAY BE INCORRECT. ALWAYS CONSULT A QUALIFIED DERMATOLOGIST OR "
    "LICENSED HEALTHCARE PROFESSIONAL BEFORE MAKING ANY DECISIONS ABOUT YOUR HEALTH. "
    "DO NOT DELAY SEEKING PROFESSIONAL MEDICAL CARE BASED ON ANYTHING THIS APP TELLS YOU. "
    "IF YOU ARE EXPERIENCING A MEDICAL EMERGENCY, CONTACT YOUR LOCAL EMERGENCY SERVICES IMMEDIATELY."
)

# ── Load DermLIP ──────────────────────────────────────────────────────────────
print("Downloading DermLIP weights...")
checkpoint_path = hf_hub_download(
    repo_id="redlessone/DermLIP_ViT-B-16",
    filename="open_clip_pytorch_model.bin"
)

print("Loading DermLIP model...")
# Create model architecture without pretrained weights first
# (Derm1M's factory.py has a missing import bug in the pretrained loading path)
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-B-16",
)
# Load DermLIP checkpoint weights separately
open_clip.load_checkpoint(model, checkpoint_path)
tokenizer = open_clip.get_tokenizer("ViT-B-16")
model = model.to(DEVICE)
model.eval()
print("DermLIP ready!")

# ── Comprehensive skin conditions list ───────────────────────────────────────
# Sources: Global dermatology + Cameroon/West Africa specific studies

# -- Load skin conditions from external file ----------------------------------
# UPGRADE PATH: Edit conditions.txt freely - no Python changes needed.
# In Docker the file is mounted as a volume, so update + restart = immediate effect.
_CONDITIONS_FILE = "/app/conditions.txt"


def _load_conditions(path: str) -> list[str]:
    """Load skin condition descriptors from a text file.
    Lines starting with # or blank lines are ignored.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            conditions = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
        print(f"Loaded {len(conditions)} skin conditions from {path}")
        return conditions
    except FileNotFoundError:
        print(f"WARNING: {path} not found -- no conditions loaded!")
        return []

def _load_clinical_kb(path: str) -> dict:
    """Load the deterministic clinical knowledge base."""
    try:
        import json
        with open(path, "r", encoding="utf-8") as f:
            kb = json.load(f)
            
            # Startup validation checks
            required_keys = [
                "condition_id", "display_name", "category", 
                "description", "general_management", "emergency_symptoms"
            ]
            for key, entry in kb.items():
                if key.startswith("_"):
                    continue
                for req in required_keys:
                    if req not in entry:
                        raise ValueError(f"KB validation failed: Entry '{key}' missing required field '{req}'.")
                        
            print(f"Loaded {len(kb)} KB entries from {path} (Schema Validated)")
            return kb
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")
        # Fail hard if KB is missing or corrupt to prevent silent runtime failures
        raise RuntimeError(f"Critical System Failure: Unable to load Clinical KB. {e}")

SKIN_CONDITIONS = _load_conditions(_CONDITIONS_FILE)
CLINICAL_KB = _load_clinical_kb("/app/clinical_kb.json")

# ── Phase 3: Inject Background 'Noise Drain' Class ─────────────────────────────
# This prevents CLIP from being forced to pick a disease for blurry/normal skin
_NORMAL_SKIN_KEY = "normal_skin_healthy_tissue"
if _NORMAL_SKIN_KEY not in SKIN_CONDITIONS:
    SKIN_CONDITIONS.append(_NORMAL_SKIN_KEY)
    CLINICAL_KB[_NORMAL_SKIN_KEY] = {
        "condition_id": _NORMAL_SKIN_KEY,
        "display_name": "Normal Skin / Healthy Tissue",
        "description": "healthy normal skin with no lesions or signs of disease",
        "category": "other", "risk_level": "low", "allow_self_care": True,
        "emergency_symptoms": [], "general_management": "Use sunscreen and moisturizer."
    }

# ── Hierarchical Category Map ────────────────────────────────────────────────
# Maps condition_key -> super-category string (built by build_hierarchy.py)
def _load_hierarchy_map(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            hmap = json.load(f)
        print(f"Loaded hierarchy map with {len(hmap)} entries")
        return hmap
    except Exception as e:
        print(f"WARNING: Could not load hierarchy map: {e}. Falling back to flat search.")
        return {}

HIERARCHY_MAP = _load_hierarchy_map("/app/hierarchy_map.json")

# Build reverse map: super-category -> list of (condition_string, condition_key)
# The condition strings here are the full SKIN_CONDITIONS entries
def _build_category_buckets(conditions: list, hmap: dict) -> dict:
    """Group full condition strings by their super-category."""
    buckets = {}
    for cond_str in conditions:
        # Extract normalized key (same logic as KB lookup)
        raw_name = cond_str.split(" — ")[0].strip() if " — " in cond_str else cond_str.strip()
        key = re.sub(r'[^a-z0-9_]', '', raw_name.replace(" ", "_").replace("-", "_").lower())
        cat = hmap.get(key, "Other General Dermatosis")
        if cat not in buckets:
            buckets[cat] = []
        buckets[cat].append(cond_str)
    return buckets

CATEGORY_BUCKETS = _build_category_buckets(SKIN_CONDITIONS, HIERARCHY_MAP)
CATEGORY_NAMES   = sorted(CATEGORY_BUCKETS.keys())

# ── Optimized Feature Cache (Research-Backed) ────────────────────────────────
# Caching text features at startup gives a ~15x classification speedup.
SOFT_CATEGORY_BONUS = 2.0  # Additive logit nudge for hierarchical alignment
HARD_LOCK_CONFIDENCE_THRESHOLD = 0.7 # Confidence threshold to lock-in a category


def _precompute_text_features(model, tokenizer, conditions, kb, categories):
    print(f"Phase 3: Pre-computing knowledge-rich features for {len(conditions)} conditions...")
    
    with torch.no_grad():
        all_condition_features = []
        
        for cond_str in conditions:
            # 1. Look up clinical details for this condition
            # Extract key (e.g. "acne" from "acne — typical presentation...")
            raw_name = cond_str.split(" — ")[0].strip() if " — " in cond_str else cond_str.strip()
            key = re.sub(r'[^a-z0-9_]', '', raw_name.replace(" ", "_").replace("-", "_").lower())
            
            entry = kb.get(key, {})
            desc  = entry.get("description", "a clinical skin condition")
            name  = entry.get("display_name", raw_name)
            
            # 2. Build Knowledge-Rich Prompts
            # We use 7 high-quality descriptive prompts instead of 5.
            # Research shows quality and specificity > quantity for large vocabularies.
            prompts = [
                f"a clinical photo of {name}, characterized by {desc.lower()}",
                f"a close-up dermatology image showing {name}: {desc.lower()}",
                f"an image of {name}, a {entry.get('category', 'skin condition')}",
                f"a medical photograph of {name} on the skin",
                f"{name}: {desc}", # Pure descriptor
                f"The skin lesion depicted is {name}, which is described as {desc.lower()}",
                f"This image depicts {name}, a type of skin condition. The description is: {desc.lower()}"
            ]
            
            # 3. Encode and ensemble
            text_inputs = tokenizer(prompts).to(DEVICE)
            feats = model.encode_text(text_inputs)
            feats /= feats.norm(dim=-1, keepdim=True)
            
            # Weighted Average: Give the detailed clinical description 2x weight
            weights = torch.tensor([1.5, 1.5, 1.0, 1.0, 2.0, 1.5, 1.5], device=DEVICE).view(-1, 1)
            ensembled_feat = (feats * weights).sum(dim=0, keepdim=True)
            ensembled_feat /= ensembled_feat.norm(dim=-1, keepdim=True)
            
            all_condition_features.append(ensembled_feat)
        
        condition_features = torch.cat(all_condition_features, dim=0) # [N, D]

        # 4. Category Features (For Pass 1 soft-locking)
        # Final shape: [N_CATEGORIES, D]
        cat_prompts = [f"a clinical photo of a skin condition related to {cat.lower()}" for cat in categories]
        cat_inputs = tokenizer(cat_prompts).to(DEVICE)
        category_features = model.encode_text(cat_inputs)
        category_features /= category_features.norm(dim=-1, keepdim=True)
        
        return condition_features, category_features

# Perform caching once at startup
CACHED_CONDITION_FEATURES, CACHED_CATEGORY_FEATURES = _precompute_text_features(
    model, tokenizer, SKIN_CONDITIONS, CLINICAL_KB, CATEGORY_NAMES
)
print("Phase 3: Text feature caching complete!")


# ── Display name: extract first 2-3 meaningful words from the condition string ──
def clean_display_name(condition: str) -> str:
    """Turn 'acne vulgaris common acne ...' into 'Acne Vulgaris'."""
    # Remove separators and special characters before splitting
    clean_raw = condition.replace("—", " ").replace("-", " ").replace("|", " ")
    words = clean_raw.split()
    
    stops = {"skin", "infection", "fungal", "viral", "bacterial", "rash", "disease", "condition"}
    keep = []
    for i, w in enumerate(words):
        if i < 2:
            keep.append(w)
        elif i == 2 and w.lower() not in stops and len(w) > 1:
            keep.append(w)
        else:
            break
    return " ".join(keep).strip().title()


# ── Deterministic Templates & Prompts ─────────────────────────────────────────

def build_raw_template(kb_entry: dict) -> str:
    """Builds the deterministic clinical string."""
    return f"""Condition: {kb_entry.get('display_name', 'Unknown')}

Description:
{kb_entry.get('description', 'No description available.')}

Common causes:
{kb_entry.get('common_causes', 'N/A')}

General management:
{kb_entry.get('general_management', 'Consult a healthcare professional.')}

Seek medical care if:
{kb_entry.get('when_to_seek_medical_care', 'Symptoms worsen or do not improve.')}"""

def build_polish_prompt(raw_template: str) -> str:
    """Instructs the LLM to strictly format the deterministic text."""
    return f"""<start_of_turn>user
Rewrite the following medical explanation to be more clear, professional, and patient-friendly.
- IMPORTANT: You MUST keep the condition name '{raw_template.splitlines()[0]}' exactly as written.
- Maintain all medical facts, causes, and management steps.
- Do not add any new medical information, warnings, or personal opinions.
- Return only the rewritten plain text.

Content to rewrite:
{raw_template}<end_of_turn>
<start_of_turn>model
"""


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "llm": OLLAMA_MODEL}


@app.post("/classify")
async def classify(file: UploadFile = File(...), skip_llm: bool = False):
    """
    Classify a skin condition from an uploaded image.
    
    Args:
        file: The image file to classify.
        skip_llm: If True, bypass the LLM polish step entirely. Use this during
                  automated evaluation runs for a massive speed boost (~30x faster
                  per image). Classification accuracy, KB data, and all metrics
                  are still fully computed and returned.
    """

    # 1. Load and preprocess image
    image_bytes = await file.read()
    base_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 1b. Test-Time Augmentation (TTA) - Generate 5 robust variations
    aug_images = [
        base_image,                                                              # Original
        base_image.transpose(Image.FLIP_LEFT_RIGHT),                             # Flipped
        ImageEnhance.Brightness(base_image).enhance(1.15),                       # Brighter
        ImageEnhance.Contrast(base_image).enhance(1.15),                         # High Contrast
        base_image.rotate(15, resample=Image.BICUBIC, fillcolor=(0,0,0))         # Rotated
    ]
    
    # Preprocess all 5 variants into a single batched tensor
    image_tensors = torch.cat([preprocess(img).unsqueeze(0) for img in aug_images]).to(DEVICE)

    # 2. Run DermLIP classification — Optimized Soft-Hierarchical Pipeline
    with torch.no_grad():
        # Get learned logit scale from the model
        logit_scale = model.logit_scale.exp()
        
        # Encode all 5 TTA image variants once
        image_features = model.encode_image(image_tensors)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        # Average TTA variants → single representative image vector
        avg_image_features = image_features.mean(dim=0, keepdim=True)  # [1, D]

        # ── PASS 1: Super-Category Prediction (Soft Nudge) ────────────────────
        # Identify which disease family the image likely belongs to.
        category_scores = {}
        locked_category = None
        locked_cat_confidence = 0.0

        if CATEGORY_NAMES:
            # DOT PRODUCT vs CACHED CATEGORY DESCRIPTORS
            cat_logits = (logit_scale * avg_image_features @ CACHED_CATEGORY_FEATURES.T).softmax(dim=-1)
            for cat, score in zip(CATEGORY_NAMES, cat_logits[0].tolist()):
                category_scores[cat] = score

            locked_category = max(category_scores, key=category_scores.get)
            locked_cat_confidence = category_scores[locked_category]
            print(f"[Soft-Hierarchy] Category guess: '{locked_category}' ({locked_cat_confidence*100:.1f}%)")

        # ── PASS 2: Fine-Grained Classification (Full Vocabulary) ──────────────
        
        # Calculate raw logits against all 277 conditions
        # (Average TTA image features vs pre-computed condition features)
        raw_logits = (logit_scale * avg_image_features @ CACHED_CONDITION_FEATURES.T) # [1, 277]
        
        # Apply Soft Category Bonus or Hard Lock
        if locked_category:
            is_hard_locked = locked_cat_confidence > HARD_LOCK_CONFIDENCE_THRESHOLD
            
            if is_hard_locked:
                # HARD LOCK: Zero out logits for all conditions outside the locked category
                print(f"[Hard-Lock] Confidence > {HARD_LOCK_CONFIDENCE_THRESHOLD*100}%. Locking to '{locked_category}'.")
                mask = torch.zeros_like(raw_logits)
                for i, cond_str in enumerate(SKIN_CONDITIONS):
                    raw_name = cond_str.split(" — ")[0].strip() if " — " in cond_str else cond_str.strip()
                    key = re.sub(r'[^a-z0-9_]', '', raw_name.replace(" ", "_").replace("-", "_").lower())
                    if HIERARCHY_MAP.get(key) == locked_category:
                        mask[0, i] = 1.0
                raw_logits *= mask
                # Set non-mask logits to a very low number to ensure they get near-zero probability
                raw_logits[mask == 0] = -1e9

            else:
                # SOFT BONUS: Nudge conditions in the predicted category higher
                for i, cond_str in enumerate(SKIN_CONDITIONS):
                    raw_name = cond_str.split(" — ")[0].strip() if " — " in cond_str else cond_str.strip()
                    key = re.sub(r'[^a-z0-9_]', '', raw_name.replace(" ", "_").replace("-", "_").lower())
                    if HIERARCHY_MAP.get(key) == locked_category:
                        raw_logits[0, i] += SOFT_CATEGORY_BONUS
                print(f"[Soft-Hierarchy] Applied +{SOFT_CATEGORY_BONUS} bonus to {len(CATEGORY_BUCKETS[locked_category])} matches.")


        # Final Softmax to get probabilities
        probs = raw_logits.softmax(dim=-1)[0].tolist()

    ranked = sorted(
        zip(SKIN_CONDITIONS, probs),
        key=lambda x: x[1],
        reverse=True
    )

    top_condition = ranked[0][0]
    top_confidence = ranked[0][1] * 100
    display_name = clean_display_name(top_condition)

    # Ensure ranked list provides possibilities

    alternatives = [
        {"condition": clean_display_name(c), "confidence": f"{p * 100:.1f}%"}
        for c, p in ranked[1:6]
    ]

    # 3. Deterministic Knowledge Base Lookup
    # Extract the name from the full condition string (before the ' — ')
    raw_name = top_condition.split(" — ")[0].strip() if " — " in top_condition else top_condition.strip()
    
    # Normalize classification string aggressively before searching KB
    search_key = re.sub(r'[^a-z0-9_]', '', raw_name.replace(" ", "_").replace("-", "_").lower())
    kb_entry = CLINICAL_KB.get(search_key)
    
    if not kb_entry:
        print(f"No specific clinical KB entry for '{search_key}', using generic_fallback")
        kb_entry = CLINICAL_KB.get("generic_fallback", {
            "display_name": display_name,
            "risk_level": "unknown",
            "allow_self_care": False,
            "emergency_symptoms": ["Seek immediate care if symptoms spread rapidly or cause severe pain."],
            "confidence_guidance": "Consult a doctor for an accurate diagnosis."
        })
    
    # Sync display_name with the official KB name (source of truth for validation)
    display_name = kb_entry.get("display_name", display_name)

    # Prepare deterministic output
    raw_template = build_raw_template(kb_entry)
    emergency_list = kb_entry.get("emergency_symptoms", ["Consult a doctor immediately if symptoms worsen."])
    clinical_severity = kb_entry.get("risk_level", "unknown")
    allow_self_care = kb_entry.get("allow_self_care", False)

    # 4. Optional LLM Polish (Strictly constrained)
    # Skip entirely when skip_llm=True (e.g. during automated evaluation benchmarking)
    llm_text = raw_template
    failure_reason = "skipped" if skip_llm else "N/A"

    if skip_llm:
        pass  # Use raw_template directly — no LLM call
    else:
        prompt = build_polish_prompt(raw_template)
        try:
            async with httpx.AsyncClient(timeout=150.0) as client:
                response = await client.post(
                    OLLAMA_GENERATE_URL,
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,  # Ultra-stable for medical rewriting
                            "top_p": 0.9,
                            "top_k": 40,
                            "repeat_penalty": 1.2,
                            "num_predict": 400,
                            "stop": ["<start_of_turn>", "<end_of_turn>", "User:", "Assistant:"]
                        }
                    }
                )
                response.raise_for_status()
                polished_text = response.json().get("response", "").strip()
                
                # Check if at least the first two major words of the diagnosis are preserved
                name_keywords = [w.lower() for w in display_name.split() if w.lower() not in {"and", "the", "with"}][:2]
                missing_keywords = [w for w in name_keywords if w not in polished_text.lower()]
                is_valid = True

                if "```" in polished_text:
                    is_valid = False
                    failure_reason = "Markdown blocks detected"
                elif len(polished_text) < 100:
                    is_valid = False
                    failure_reason = f"Output too short ({len(polished_text)} chars)"
                elif missing_keywords:
                    is_valid = False
                    failure_reason = f"Diagnosis drift: '{' '.join(missing_keywords)}' missing"
                # NOTE: Length deviation check intentionally removed (Option A).
                # Gemma naturally elaborates or condenses when rephrasing medical text,
                # and a character-count threshold was rejecting ~100% of valid outputs.
                # Safety is maintained by the diagnosis keyword drift check above.

                if is_valid:
                    llm_text = polished_text.replace("*", "").replace("#", "")
                    llm_text = re.sub(r'(\b\w+\b)(?:\W+\1){2,}', r'\1', llm_text, flags=re.IGNORECASE)
                    lines = llm_text.split('\n')
                    unique_lines = []
                    seen = set()
                    for line in lines:
                        line_lower = line.lower().strip()
                        if line_lower and line_lower not in seen:
                            unique_lines.append(line)
                            seen.add(line_lower)
                        elif not line_lower:
                            unique_lines.append(line)
                    llm_text = '\n'.join(unique_lines).strip()
                else:
                    print(f"LLM polish REJECTED: {failure_reason}. Using raw template.")
        except Exception as e:
            print(f"LLM styling failed: {type(e).__name__}: {e}. Falling back to raw template.")
            llm_text = raw_template
        
    # Dynamic confidence rule engine
    if top_confidence >= 75.0:
        confidence_prefix = "This prediction has high confidence."
    elif top_confidence >= 50.0:
        confidence_prefix = "This prediction has moderate confidence."
    else:
        confidence_prefix = "The model's confidence is limited, and alternative conditions are possible."
        
    llm_text = f"{confidence_prefix}\n\n{llm_text}"

    # 4. Store session for follow-up chat
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "condition": display_name,
        "confidence": f"{top_confidence:.1f}%",
        "explanation": llm_text,
        "allow_self_care": allow_self_care,
        "chat_history": [],  # list of {"role": "user"|"assistant", "content": str}
    }

    # 5. Return full structured response
    response_body = {
        "session_id": session_id,  # pass this back with /chat requests
        "disclaimer": DISCLAIMER,
        "classification": {
            "top_prediction": display_name,
            "confidence": f"{top_confidence:.1f}%",
            "clinical_severity": clinical_severity,
            "hierarchical_category": locked_category or "N/A",
            "category_confidence": f"{locked_cat_confidence * 100:.1f}%" if locked_category else "N/A",
            "other_possibilities": alternatives,
        },
        "explanation": llm_text,
        "raw_explanation": raw_template,
        "seek_emergency_care_immediately_if": emergency_list
    }

    return JSONResponse(response_body)

# ── Chat endpoint ───────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.post("/chat")
async def chat(req: ChatRequest):
    session = SESSIONS.get(req.session_id)
    if not session:
        return JSONResponse(
            {"error": "Session not found. Please classify an image first."},
            status_code=404
        )

    # Build messages for /api/chat - Ollama applies the correct chat template automatically
    self_care_directive = (
        "You may suggest general skincare advice."
        if session.get('allow_self_care', False)
        else "Advise the patient to see a doctor for treatment options."
    )

    system_content = (
        f"You are a dermatology assistant. "
        f"The patient has: {session['condition']}. "
        f"Give a brief, factual answer. {self_care_directive}"
    )

    messages = [{"role": "system", "content": system_content}]

    # Append recent history (last 6 turns)
    for turn in session["chat_history"][-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": req.message})

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                OLLAMA_CHAT_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 50,
                        "repeat_penalty": 1.3,
                        "num_predict": 300,
                        "stop": ["<end_of_turn>", "\n\nPatient:", "\n\nQuestion:"]
                    }
                }
            )
            response.raise_for_status()
            reply = response.json().get("message", {}).get("content", "").strip()
            print(f"\n--- RAW CHAT REPLY ---\n{reply[:300]}\n----------------------\n")

            # --- Loop Detection ---
            def _is_looping(text: str) -> bool:
                # 1. Any substring of 2-20 chars repeating 4+ times in a row
                #    Catches: "I-I-I-I", "word-word-word", "ha ha ha ha"
                if re.search(r'(.{2,20})\1{3,}', text, re.IGNORECASE):
                    return True
                # 2. Token uniqueness ratio — collapse means vocab shrinks to near-zero
                tokens = re.findall(r'\b\w+\b', text.lower())
                if len(tokens) > 10 and len(set(tokens)) / len(tokens) < 0.25:
                    return True
                # 3. Sentence-level: >40% of sentences are duplicates
                sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 8]
                if len(sentences) > 2 and len(set(s.lower() for s in sentences)) < len(sentences) * 0.6:
                    return True
                return False

            if _is_looping(reply):
                print(f"LOOP DETECTED in response: {reply[:100]}...")
                reply = "I'm having trouble generating a proper response. Please rephrase your question or consult a dermatologist directly."
            else:
                # Remove markdown artifacts
                reply = reply.replace("*", "").replace("#", "")
                # Clean up trailing fragments
                reply = re.sub(r'\n[A-Za-z]{1,5}$', '', reply)
                reply = re.sub(r'[,:]$', '.', reply)
            
            # Final validation
            if len(reply) < 20:
                reply = "Please consult a qualified healthcare professional for personalized medical advice regarding your condition."
            
    except Exception as e:
        print(f"Chat LLM error: {type(e).__name__}: {e}")
        reply = "I'm unable to generate a response right now. Please consult a healthcare professional for medical advice."

    # Save to history for multi-turn continuity
    session["chat_history"].append({"role": "user", "content": req.message})
    session["chat_history"].append({"role": "assistant", "content": reply})

    return JSONResponse({
        "session_id": req.session_id,
        "reply": reply,
        "disclaimer": DISCLAIMER,
    })
