"""
evaluate.py  —  DermLIP Server Evaluation Suite
================================================
Runs four evaluation modules against a running DermLIP server.

USAGE
-----
  # Full evaluation (all modules)
  python evaluate.py --images "C:/Users/HP x360 1030 G2/Desktop/TEST PROD/image data"

  # Only classification accuracy
  python evaluate.py --images "..." --test classify

  # Only latency (needs at least one image)
  python evaluate.py --images "..." --test latency --n 20

  # Only KB coverage (no images needed)
  python evaluate.py --test kb

  # Only chat quality (auto-creates a session from first available image)
  python evaluate.py --images "..." --test chat

  # Change server URL (e.g. running on another machine)
  python evaluate.py --images "..." --url http://192.168.1.10:8000

EXPECTED IMAGE-DIRECTORY LAYOUT
--------------------------------
  image data/
      acne_vulgaris/        <-- folder name = ground-truth label
          01.jpg
          02.jpg
      rosacea/
          01.jpg
      ...

REQUIREMENTS
------------
  pip install requests
  (All other imports are from the Python standard library.)

REPORTS
-------
  Each module writes a timestamped JSON + CSV into ./eval_reports/
"""

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    import requests
except ImportError:
    print("[ERROR] 'requests' is not installed.  Run:  pip install requests")
    sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  —  edit these if your setup differs
# =════════════════════════════════════════════════════════════════════════════

BASE_URL        = "http://localhost:8000"   # DermLIP server address
KB_PATH         = "clinical_kb.json"        # Path to the KB file (relative to this script)
CONDITIONS_PATH = "conditions.txt"          # Path to conditions list (relative to this script)
REPORT_DIR      = "eval_reports"            # Output folder for JSON / CSV reports
REQUEST_TIMEOUT = 180                       # Per-request HTTP timeout in seconds

# Questions sent to /chat during the chat-quality module.
# Covers contagion, self-care, causes, urgency — the most common patient queries.
CHAT_QUESTIONS: List[str] = [
    "Is this condition contagious?",
    "Can I treat this at home with an over-the-counter cream?",
    "What are the main causes of this condition?",
    "How long does it usually take to heal?",
    "What should I avoid doing while I have this?",
    "When should I go to the emergency room?",
    "Can this condition spread to other parts of my body?",
    "Is this condition linked to my diet or stress?",
]

# Words that are too generic to be useful when fuzzy-matching folder names
# against predicted condition names (e.g. "skin", "infection", "type").
FUZZY_STOP_WORDS = {
    "skin", "infection", "disease", "condition", "type",
    "form", "acute", "chronic", "common", "other",
}


# ═════════════════════════════════════════════════════════════════════════════
# TERMINAL HELPERS  —  colours + simple print wrappers
# =════════════════════════════════════════════════════════════════════════════

def _ansi_supported() -> bool:
    """Return True when the terminal is likely to render ANSI escape codes."""
    return sys.platform != "win32" or "WT_SESSION" in os.environ


_ANSI = _ansi_supported()

# Symbols shown next to each result line
SYM_OK   = "\033[92m✔\033[0m" if _ANSI else "[OK]  "
SYM_WARN = "\033[93m⚠\033[0m" if _ANSI else "[WARN]"
SYM_ERR  = "\033[91m✖\033[0m" if _ANSI else "[ERR] "
SYM_INFO = "\033[96m•\033[0m" if _ANSI else "[INFO]"
BOLD     = "\033[1m"           if _ANSI else ""
RESET    = "\033[0m"           if _ANSI else ""


def section(title: str) -> None:
    """Print a bold section header with a horizontal rule."""
    rule = "─" * 62
    print(f"\n{BOLD}{rule}\n  {title}\n{rule}{RESET}")


def log_ok(msg: str)   -> None: print(f"  {SYM_OK}   {msg}")
def log_warn(msg: str) -> None: print(f"  {SYM_WARN} {msg}")
def log_err(msg: str)  -> None: print(f"  {SYM_ERR}  {msg}")
def log_info(msg: str) -> None: print(f"  {SYM_INFO} {msg}")


# ═════════════════════════════════════════════════════════════════════════════
# SHARED API HELPERS
# =════════════════════════════════════════════════════════════════════════════

def api_classify(image_path: str, skip_llm: bool = False) -> Optional[Dict[str, Any]]:
    """
    POST an image file to /classify.
    Returns the parsed JSON response dict, or None on any error.
    Pass skip_llm=True to bypass the LLM polish step on the server
    for a massive speed boost during evaluation runs.
    """
    try:
        with open(image_path, "rb") as fh:
            resp = requests.post(
                f"{BASE_URL}/classify",
                files={"file": fh},
                params={"skip_llm": "true"} if skip_llm else None,
                timeout=REQUEST_TIMEOUT,
            )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        # Surface the error without crashing the whole run
        log_err(f"classify failed [{Path(image_path).name}]: {exc}")
        return None


def api_chat(session_id: str, message: str) -> Optional[Dict[str, Any]]:
    """
    POST a follow-up message to /chat for a given session.
    Returns the parsed JSON response dict, or None on any error.
    """
    try:
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"session_id": session_id, "message": message},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        log_err(f"chat failed: {exc}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# SHARED UTILITY FUNCTIONS
# =════════════════════════════════════════════════════════════════════════════

def normalise(text: str) -> str:
    """Lowercase and collapse all punctuation/underscores to spaces."""
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).strip()


def folder_matches_prediction(folder_name: str, predicted: str) -> bool:
    """
    Fuzzy label matcher.  Converts a folder name like 'acne_vulgaris' into
    meaningful words, then checks what fraction of those words appear anywhere
    in the predicted condition string.

    ≥50 % word overlap is treated as a match.

    Examples that pass:
        'acne_vulgaris'   vs  'Acne Inflammatory Papulopustular'  → True
        'rosacea'         vs  'Rosacea Papulopustular'             → True
        'tinea_corporis'  vs  'Tinea Corporis Ringworm'            → True
    """
    pred_norm   = normalise(predicted)
    folder_norm = normalise(folder_name.replace("_", " ").replace("-", " "))

    # Filter out generic stop-words so we don't get false positives
    key_words = [
        w for w in folder_norm.split()
        if len(w) > 3 and w not in FUZZY_STOP_WORDS
    ]
    if not key_words:
        return False  # nothing meaningful to match against

    hit_count = sum(1 for w in key_words if w in pred_norm)
    return hit_count / len(key_words) >= 0.5


def normalise_to_kb_key(condition_str: str) -> str:
    """
    Mirror the key-normalisation logic used in app.py so we can check KB
    coverage offline without calling the server.

    'Acne Inflammatory Papulopustular — inflamed papules' → 'acne_inflammatory_papulopustular'
    """
    # Strip the '— description' suffix when present
    raw = condition_str.split(" — ")[0].strip() if " — " in condition_str else condition_str.strip()
    # Remove every character that isn't alphanumeric or underscore
    return re.sub(r"[^a-z0-9_]", "", raw.replace(" ", "_").replace("-", "_").lower())


def is_looping(text: str) -> bool:
    """
    Offline duplicate of app.py's _is_looping() guard.
    Returns True when the text shows signs of autoregressive collapse:
        1. Any substring of 2-20 chars that repeats 4+ times consecutively.
        2. Unique-token ratio below 25 % (vocabulary collapse).
        3. More than 40 % of distinct sentences are duplicates.
    """
    # ── Check 1: raw substring repetition ─────────────────────────────────
    if re.search(r"(.{2,20})\1{3,}", text, re.IGNORECASE):
        return True

    # ── Check 2: vocabulary collapse ──────────────────────────────────────
    tokens = re.findall(r"\b\w+\b", text.lower())
    if len(tokens) > 10 and len(set(tokens)) / len(tokens) < 0.25:
        return True

    # ── Check 3: sentence-level repetition ────────────────────────────────
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if len(s.strip()) > 8]
    if (
        len(sentences) > 2
        and len({s.lower() for s in sentences}) < len(sentences) * 0.6
    ):
        return True

    return False


def percentile(data: List[float], pct: float) -> float:
    """
    Compute a percentile value without NumPy.
    Uses linear interpolation between the two nearest ranks.
    """
    if not data:
        return 0.0
    s  = sorted(data)
    k  = (len(s) - 1) * pct / 100
    lo = int(k)
    hi = lo + 1
    fr = k - lo
    if hi >= len(s):
        return s[lo]
    return s[lo] + fr * (s[hi] - s[lo])


def save_report(name: str, data: Any) -> str:
    """
    Save `data` to REPORT_DIR as JSON (always) and CSV (when data is a list
    of dicts).  Returns a human-readable description of the files written.
    """
    Path(REPORT_DIR).mkdir(exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{REPORT_DIR}/{name}_{ts}"

    # ── JSON ──────────────────────────────────────────────────────────────
    json_path = f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

    # ── CSV (only for flat lists of dicts) ────────────────────────────────
    if isinstance(data, list) and data and isinstance(data[0], dict):
        csv_path = f"{stem}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        return f"{json_path}  +  {csv_path}"

    return json_path


def first_image_in(root: str) -> Optional[str]:
    """Return the path of the first image file found recursively under root."""
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        hits = list(Path(root).rglob(ext))
        if hits:
            return str(hits[0])
    return None


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 0  —  SERVER HEALTH CHECK
# =════════════════════════════════════════════════════════════════════════════

def run_health() -> bool:
    """
    Ping /health.  Returns True if the server is up and responsive.
    This is always run before any other module.
    """
    section("SERVER HEALTH")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        log_ok(f"Status : {data.get('status', '?')}")
        log_ok(f"Device : {data.get('device', '?')}")
        log_ok(f"LLM    : {data.get('llm', '?')}")
        return True
    except Exception as exc:
        log_err(f"Server unreachable at {BASE_URL} — {exc}")
        log_err("Make sure the DermLIP container is running.")
        return False


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 1  —  CLASSIFICATION ACCURACY
# =════════════════════════════════════════════════════════════════════════════

def run_classify(images_dir: str, top_k: int = 5, workers: int = 1, skip_llm: bool = False) -> Dict[str, Any]:
    """
    Walk `images_dir`.  Every sub-folder is treated as a ground-truth label.
    For each image inside a sub-folder, POST to /classify and compare the
    returned predictions against the folder label using fuzzy matching.

    Metrics computed
    ────────────────
    • Top-1 accuracy   — is the single top prediction correct?
    • Top-k accuracy   — is the correct label anywhere in the top-k results?
    • KB hit rate      — did the top prediction resolve to a KB entry
                         (or fall back to generic_fallback)?
    • LLM polish rate  — did the LLM successfully rewrite the template
                         (explanation ≠ raw_explanation)?
    • Per-label counts — broken down by condition folder
    • Average confidence of top predictions
    • Average latency per image

    Performance Options
    ───────────────────
    • workers   — number of parallel threads (default: 1). Set to 4 for ~4x speedup.
    • skip_llm  — bypass the LLM polish on the server (default: False). This alone
                  gives a ~20-40x speedup per image. All accuracy metrics are still
                  fully computed; only the cosmetic rewriting is skipped.
    """
    section("MODULE 1  —  CLASSIFICATION ACCURACY")

    images_root = Path(images_dir)
    if not images_root.exists():
        log_err(f"Directory not found: {images_dir}")
        return {}

    # Gather all sub-folders (each = one ground-truth class)
    label_dirs = sorted([d for d in images_root.iterdir() if d.is_dir()])
    if not label_dirs:
        log_err("No sub-folders found inside the images directory.")
        log_err("Expected layout:  images_dir/<label_name>/<image.jpg>")
        return {}

    log_info(f"Found {len(label_dirs)} label folders in: {images_dir}\n")

    # Accumulators
    rows: List[Dict] = []          # one row per image — written to CSV
    total     = 0
    top1_hits = 0
    topk_hits = 0
    kb_hits   = 0
    polish_ok = 0
    safety_rejected = 0
    tone_mismatches = 0

    for label_dir in label_dirs:
        label     = label_dir.name
        img_files = (
            list(label_dir.glob("*.jpg"))  +
            list(label_dir.glob("*.jpeg")) +
            list(label_dir.glob("*.png"))  +
            list(label_dir.glob("*.bmp"))
        )
        if not img_files:
            log_warn(f"Skipping empty folder: {label}")
            continue

        lbl_top1 = lbl_topk = 0

        # ── Process images (parallel or sequential) ───────────────────────
        def _process_one(img_path: Path) -> Dict:
            """Classify one image and return a result dict."""
            t0   = time.perf_counter()
            resp = api_classify(str(img_path), skip_llm=skip_llm)
            ms   = (time.perf_counter() - t0) * 1000
            if resp is None:
                return {"image": img_path.name, "label": label, "status": "api_error"}

            clf          = resp.get("classification", {})
            top_pred     = clf.get("top_prediction", "")
            others       = [o.get("condition", "") for o in clf.get("other_possibilities", [])]
            raw_conf     = float(clf.get("confidence", "0%").replace("%", ""))
            clinical_sev = clf.get("clinical_severity", "unknown")
            raw_expl     = resp.get("raw_explanation", "")
            has_kb       = bool(raw_expl) and "No description available" not in raw_expl
            explanation  = resp.get("explanation", "")
            prefix_line  = explanation.splitlines()[0] if explanation else ""
            expl_body    = "\n".join(explanation.splitlines()[2:]).strip() if "\n" in explanation else ""
            was_rejected = (expl_body.strip() == raw_expl.strip()) and has_kb
            was_polished = not was_rejected and len(expl_body) > 50

            if raw_conf >= 75.0:   expected_tone = "high confidence"
            elif raw_conf >= 50.0: expected_tone = "moderate confidence"
            else:                  expected_tone = "confidence is limited"
            tone_match = expected_tone in prefix_line.lower()

            t1 = folder_matches_prediction(label, top_pred)
            all_preds = [top_pred] + others[:top_k - 1]
            tk = any(folder_matches_prediction(label, p) for p in all_preds)

            return {
                "image"       : img_path.name,
                "label"       : label,
                "top_pred"    : top_pred,
                "severity"    : clinical_sev,
                "confidence"  : f"{raw_conf:.1f}%",
                "top1_correct": t1,
                "topk_correct": tk,
                "kb_hit"      : has_kb,
                "llm_polished": was_polished,
                "safety_rej"  : was_rejected,
                "latency_ms"  : round(ms, 1),
                "tone_ok"     : tone_match,
            }

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_process_one, p): p for p in img_files}
                label_results = []
                for fut in as_completed(futures):
                    label_results.append(fut.result())
        else:
            label_results = [_process_one(p) for p in img_files]

        # Aggregate per-label results into global counters
        for r in label_results:
            total += 1
            rows.append(r)
            if r.get("top1_correct"): top1_hits += 1; lbl_top1 += 1
            if r.get("topk_correct"): topk_hits += 1; lbl_topk += 1
            if r.get("kb_hit"):       kb_hits   += 1
            if r.get("llm_polished"): polish_ok += 1
            if r.get("safety_rej"):   safety_rejected += 1
            if not r.get("tone_ok", True): tone_mismatches += 1
        n = len(img_files)
        printer = log_ok if lbl_top1 == n else (log_warn if lbl_top1 > 0 else log_err)
        printer(
            f"{label:<38}  top1={lbl_top1}/{n}  "
            f"top{top_k}={lbl_topk}/{n}"
        )

    # ── Aggregate metrics ─────────────────────────────────────────────────
    if total == 0:
        log_err("No images were processed.")
        return {}

    top1_pct    = top1_hits / total * 100
    topk_pct    = topk_hits / total * 100
    kb_pct      = kb_hits   / total * 100
    polish_pct  = polish_ok / total * 100
    reject_pct  = safety_rejected / total * 100

    confs      = [float(r["confidence"].replace("%", "")) for r in rows if "confidence" in r]
    avg_conf   = statistics.mean(confs) if confs else 0.0
    lats       = [r["latency_ms"] for r in rows if "latency_ms" in r]
    avg_lat    = statistics.mean(lats)  if lats  else 0.0

    print()
    log_ok(f"Top-1 Accuracy     : {top1_pct:.1f}%  ({top1_hits}/{total})")
    log_ok(f"Top-{top_k} Accuracy     : {topk_pct:.1f}%  ({topk_hits}/{total})")
    log_info(f"KB Hit Rate        : {kb_pct:.1f}%")
    log_info(f"LLM Polish Rate    : {polish_pct:.1f}%")
    log_info(f"Safety Reject Rate : {reject_pct:.1f}%")
    log_info(f"Tone Mismatches    : {tone_mismatches}")
    log_info(f"Avg Confidence     : {avg_conf:.1f}%")
    log_info(f"Avg Latency        : {avg_lat:.0f} ms / image")

    summary = {
        "top1_accuracy"      : round(top1_pct, 2),
        f"top{top_k}_accuracy": round(topk_pct, 2),
        "kb_hit_rate_pct"    : round(kb_pct, 2),
        "llm_polish_rate_pct": round(polish_pct, 2),
        "safety_reject_pct"  : round(reject_pct, 2),
        "tone_mismatches"    : tone_mismatches,
        "avg_confidence_pct" : round(avg_conf, 2),
        "avg_latency_ms"     : round(avg_lat, 1),
        "total_images"       : total,
        "per_image_results"  : rows,
    }
    log_info(f"Report → {save_report('classify', rows)}")
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 2  —  KNOWLEDGE BASE COVERAGE
# =════════════════════════════════════════════════════════════════════════════

def run_kb() -> Dict[str, Any]:
    """
    Cross-reference every condition in conditions.txt against clinical_kb.json.

    Any condition that has no KB entry will fall back to 'generic_fallback'
    at runtime, producing a less informative response.  This module surfaces
    those gaps so they can be filled.

    Metrics
    ───────
    • Total conditions in conditions.txt
    • KB entries present
    • Coverage percentage
    • List of uncovered conditions (printed + saved)
    """
    section("MODULE 2  —  KNOWLEDGE BASE COVERAGE")

    # ── Load conditions.txt ───────────────────────────────────────────────
    if not Path(CONDITIONS_PATH).exists():
        log_err(f"conditions.txt not found at: {CONDITIONS_PATH}")
        log_err("Run this script from the dermlip-server directory.")
        return {}

    with open(CONDITIONS_PATH, encoding="utf-8") as fh:
        conditions = [
            ln.strip() for ln in fh
            if ln.strip() and not ln.strip().startswith("#")
        ]

    # ── Load clinical_kb.json ─────────────────────────────────────────────
    if not Path(KB_PATH).exists():
        log_err(f"clinical_kb.json not found at: {KB_PATH}")
        return {}

    with open(KB_PATH, encoding="utf-8") as fh:
        kb = json.load(fh)

    # Meta-entries (prefixed with '_') are config, not conditions — exclude them
    kb_keys = {k for k in kb if not k.startswith("_")}

    # ── Match each condition to its normalised KB key ─────────────────────
    covered: List[str]   = []
    uncovered: List[str] = []

    for cond in conditions:
        key = normalise_to_kb_key(cond)
        (covered if key in kb_keys else uncovered).append(cond)

    coverage_pct = len(covered) / len(conditions) * 100 if conditions else 0.0

    log_info(f"Conditions in conditions.txt  : {len(conditions)}")
    log_info(f"Entries in clinical_kb.json   : {len(kb_keys)}")

    printer = log_ok if coverage_pct >= 80 else (log_warn if coverage_pct >= 50 else log_err)
    printer(f"Coverage                      : {coverage_pct:.1f}%  ({len(covered)}/{len(conditions)})")

    if uncovered:
        log_warn(f"\n  Conditions WITHOUT a KB entry ({len(uncovered)}):")
        # Show up to 25 in the terminal; the full list is always in the report
        for u in uncovered[:25]:
            print(f"      • {u}")
        if len(uncovered) > 25:
            log_info(f"    … and {len(uncovered) - 25} more — see the report file.")

    summary = {
        "total_conditions": len(conditions),
        "kb_entries"      : len(kb_keys),
        "covered_count"   : len(covered),
        "uncovered_count" : len(uncovered),
        "coverage_pct"    : round(coverage_pct, 2),
        "uncovered"       : uncovered,   # full list in the JSON report
    }
    log_info(f"Report → {save_report('kb_coverage', summary)}")
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 3  —  LATENCY BENCHMARK
# =════════════════════════════════════════════════════════════════════════════

def run_latency(
    sample_image: str,
    n: int       = 10,
    workers: int = 1,
) -> Dict[str, Any]:
    """
    Send `n` requests to /classify (optionally concurrently) and measure
    response-time percentiles.  Then run a single-threaded /chat benchmark
    using the session from the last successful /classify response.

    Stats reported
    ──────────────
    • min / mean / p50 / p95 / p99 / max  (milliseconds)
    • Throughput  (requests per second, concurrent mode only)
    • /chat  p50 / p95 / mean
    """
    section("MODULE 3  —  LATENCY BENCHMARK")

    if not Path(sample_image).exists():
        log_err(f"Sample image not found: {sample_image}")
        return {}

    log_info(f"Image   : {Path(sample_image).name}")
    log_info(f"Requests: {n}   Workers: {workers}")

    # ── /classify benchmark ───────────────────────────────────────────────
    classify_ms: List[float] = []
    last_session: Optional[str] = None

    def _timed_classify(_: int) -> Optional[Tuple[float, str]]:
        """Return (latency_ms, session_id) or None on failure."""
        t0   = time.perf_counter()
        resp = api_classify(sample_image)
        ms   = (time.perf_counter() - t0) * 1000
        if resp:
            return ms, resp.get("session_id", "")
        return None

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_timed_classify, i) for i in range(n)]
        for future in as_completed(futures):
            result = future.result()
            if result:
                ms, sid = result
                classify_ms.append(ms)
                last_session = sid   # save most recently returned session
    wall_elapsed = time.perf_counter() - wall_start

    if classify_ms:
        throughput = len(classify_ms) / wall_elapsed
        log_ok(
            f"/classify  "
            f"min={min(classify_ms):.0f}ms  "
            f"mean={statistics.mean(classify_ms):.0f}ms  "
            f"p50={percentile(classify_ms, 50):.0f}ms  "
            f"p95={percentile(classify_ms, 95):.0f}ms  "
            f"p99={percentile(classify_ms, 99):.0f}ms  "
            f"max={max(classify_ms):.0f}ms"
        )
        if workers > 1:
            log_info(f"           Throughput: {throughput:.2f} req/s  ({workers} workers)")
    else:
        log_err("All /classify requests failed — check server logs.")

    # ── /chat benchmark  (sequential, uses the saved session) ────────────
    chat_ms: List[float] = []

    if last_session:
        for q in CHAT_QUESTIONS:
            t0   = time.perf_counter()
            resp = api_chat(last_session, q)
            ms   = (time.perf_counter() - t0) * 1000
            if resp:
                chat_ms.append(ms)

        if chat_ms:
            log_ok(
                f"/chat      "
                f"min={min(chat_ms):.0f}ms  "
                f"mean={statistics.mean(chat_ms):.0f}ms  "
                f"p50={percentile(chat_ms, 50):.0f}ms  "
                f"p95={percentile(chat_ms, 95):.0f}ms"
            )
        else:
            log_warn("No successful /chat responses.")
    else:
        log_warn("Skipping /chat benchmark — no session available from /classify.")

    # ── Persist ───────────────────────────────────────────────────────────
    summary = {
        "classify": {
            "n"      : len(classify_ms),
            "min_ms" : round(min(classify_ms),                    1) if classify_ms else 0,
            "mean_ms": round(statistics.mean(classify_ms),        1) if classify_ms else 0,
            "p50_ms" : round(percentile(classify_ms, 50),         1),
            "p95_ms" : round(percentile(classify_ms, 95),         1),
            "p99_ms" : round(percentile(classify_ms, 99),         1),
            "max_ms" : round(max(classify_ms),                    1) if classify_ms else 0,
        },
        "chat": {
            "n"      : len(chat_ms),
            "mean_ms": round(statistics.mean(chat_ms), 1) if chat_ms else 0,
            "p50_ms" : round(percentile(chat_ms, 50),  1),
            "p95_ms" : round(percentile(chat_ms, 95),  1),
        },
    }
    log_info(f"Report → {save_report('latency', summary)}")
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 4  —  CHAT QUALITY
# =════════════════════════════════════════════════════════════════════════════

def run_chat(
    session_id   : Optional[str] = None,
    sample_image : Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send every question in CHAT_QUESTIONS to /chat and score the replies.

    Checks per response
    ───────────────────
    • Loop-free         — same logic as the in-app guard
    • Not a fallback    — model actually answered rather than triggering the error handler
    • Not truncated     — reply ends with terminal punctuation (., !, ?)
    • Minimum length    — at least 40 characters
    • Disclaimer present — the response includes the disclaimer field

    Either pass an existing --session id, or --images so a fresh session
    can be created automatically by classifying the first available image.
    """
    section("MODULE 4  —  CHAT QUALITY")

    # ── Obtain or create a session ────────────────────────────────────────
    if session_id is None:
        if sample_image and Path(sample_image).exists():
            log_info("No session_id — classifying sample image to create one …")
            resp = api_classify(sample_image)
            if resp:
                session_id = resp.get("session_id")
                condition  = resp.get("classification", {}).get("top_prediction", "?")
                log_ok(f"Session  : {session_id}")
                log_ok(f"Condition: {condition}\n")
            else:
                log_err("Failed to create a session via /classify.")
                return {}
        else:
            log_err("Provide --session <id>  or  --images <path>  for chat evaluation.")
            return {}

    rows: List[Dict] = []
    loop_count  = 0
    trunc_count = 0
    fallback_count = 0

    for q in CHAT_QUESTIONS:
        t0   = time.perf_counter()
        resp = api_chat(session_id, q)
        ms   = (time.perf_counter() - t0) * 1000

        if resp is None:
            rows.append({"question": q, "status": "api_error"})
            continue

        reply = resp.get("reply", "")

        # ── Quality checks ────────────────────────────────────────────────
        looping    = is_looping(reply)

        # Fallback message is returned when the LLM call throws an exception
        fallback   = (
            "having trouble generating" in reply.lower() or
            "unable to generate"         in reply.lower()
        )

        # Truncation: reply is long enough but ends without terminal punctuation
        truncated  = (
            len(reply) > 60 and
            not reply.rstrip().endswith((".", "!", "?", "…", '"'))
        )

        too_short  = len(reply) < 40
        has_disclaim = bool(resp.get("disclaimer"))

        # Determine overall status (worst-case wins)
        if looping or fallback:
            status = "bad"
        elif truncated or too_short:
            status = "warn"
        else:
            status = "ok"

        if looping:    loop_count     += 1
        if truncated:  trunc_count    += 1
        if fallback:   fallback_count += 1

        # Status icon for terminal
        icon = log_ok if status == "ok" else (log_warn if status == "warn" else log_err)
        icon(f"[{ms:>5.0f}ms | {len(reply):>4}ch]  {q[:55]}")

        if looping:   print("          ^ LOOP DETECTED")
        if fallback:  print("          ^ FALLBACK (LLM error)")
        if truncated: print("          ^ TRUNCATED (no terminal punctuation)")
        if too_short: print("          ^ TOO SHORT")

        rows.append({
            "question"       : q,
            "reply_length"   : len(reply),
            "latency_ms"     : round(ms, 1),
            "status"         : status,
            "looping"        : looping,
            "fallback"       : fallback,
            "truncated"      : truncated,
            "too_short"      : too_short,
            "has_disclaimer" : has_disclaim,
        })

    # ── Aggregate ─────────────────────────────────────────────────────────
    n         = len(rows)
    ok_count  = sum(1 for r in rows if r.get("status") == "ok")
    lengths   = [r["reply_length"] for r in rows if "reply_length" in r]
    lats      = [r["latency_ms"]   for r in rows if "latency_ms"   in r]
    avg_len   = statistics.mean(lengths) if lengths else 0.0
    avg_lat   = statistics.mean(lats)    if lats    else 0.0

    print()
    (log_ok if ok_count == n else log_warn)(f"OK responses     : {ok_count}/{n}")
    (log_ok if loop_count == 0     else log_err) (f"Loop rate        : {loop_count}/{n}")
    (log_ok if fallback_count == 0 else log_err) (f"Fallback rate    : {fallback_count}/{n}")
    (log_ok if trunc_count == 0    else log_warn)(f"Truncated        : {trunc_count}/{n}")
    log_info(f"Avg reply length : {avg_len:.0f} chars")
    log_info(f"Avg latency      : {avg_lat:.0f} ms")

    summary = {
        "session_id"      : session_id,
        "total_questions" : n,
        "ok_count"        : ok_count,
        "loop_count"      : loop_count,
        "fallback_count"  : fallback_count,
        "truncated_count" : trunc_count,
        "avg_reply_length": round(avg_len, 1),
        "avg_latency_ms"  : round(avg_lat, 1),
        "per_question"    : rows,
    }
    log_info(f"Report → {save_report('chat_quality', rows)}")
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 5: LLM POLISH TEST
# =════════════════════════════════════════════════════════════════════════════

def run_llm(kb_path: str = KB_PATH, n: int = 10, url: str = "http://localhost:11434/api/generate") -> Dict[str, Any]:
    """Tests the standalone LLM polish rate by formatting mock templates from the KB."""
    import random
    section("MODULE 5  —  LLM POLISH RELIABILITY")
    
    p = Path(kb_path)
    if not p.exists():
        log_err(f"{kb_path} not found.")
        return {}
        
    with open(p, "r", encoding="utf-8") as f:
        kb_data = json.load(f)
        
    entries = [e for k, e in kb_data.items() if not k.startswith("_") and k != "generic_fallback"]
    if not entries:
        log_err("No valid clinical_kb entries found.")
        return {}
        
    samples = random.sample(entries, min(n, len(entries)))
    rows = []
    accepted_count = 0
    latencies = []
    
    for i, entry in enumerate(samples, start=1):
        display_name = entry.get("display_name", "Unknown")
        print(f"\n  [INFO] ({i}/{len(samples)}) Polishing: {display_name}")
        
        # Deterministic formatting matches app.py exactly
        raw_tmpl = f"Condition: {display_name}\n\nDescription:\n{entry.get('description', '')}\n\nCommon causes:\n{entry.get('common_causes', '')}\n\nGeneral management:\n{entry.get('general_management', '')}\n\nSeek medical care if:\n{entry.get('when_to_seek_medical_care', '')}"
        
        prompt = f"<start_of_turn>user\nRewrite the following medical explanation to be more clear, professional, and patient-friendly.\n- IMPORTANT: You MUST keep the condition name '{display_name}' exactly as written.\n- Maintain all medical facts, causes, and management steps.\n- Do not add any new medical information, warnings, or personal opinions.\n- Return only the rewritten plain text.\n\nContent to rewrite:\n{raw_tmpl}<end_of_turn>\n<start_of_turn>model\n"
        
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                url,
                json={
                    "model": "gemma3n:latest",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "top_p": 0.9,
                        "top_k": 30,
                        "repeat_penalty": 1.1,
                        "num_predict": 400,
                        "stop": ["<end_of_turn>"]
                    }
                },
                timeout=90.0
            )
            resp.raise_for_status()
            polished = resp.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            log_err(f"No connection to LLM API ({url}). Are you running Ollama locally?")
            break
        except Exception as e:
            log_err(f"LLM API failed: {e}")
            continue
            
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)
        
        # Validation checks matching app.py
        name_keywords = [w.lower() for w in display_name.split() if w.lower() not in {"and", "the", "with"}][:2]
        missing_keywords = [w for w in name_keywords if w not in polished.lower()]
        
        is_valid = True
        failure_reason = None
        
        if "```" in polished:
            is_valid = False
            failure_reason = "Markdown blocks detected"
        elif len(polished) < 100:
            is_valid = False
            failure_reason = f"Output too short ({len(polished)} chars)"
        elif missing_keywords:
            is_valid = False
            failure_reason = f"Diagnosis drift: '{' '.join(missing_keywords)}' missing"
            
        if is_valid:
            accepted_count += 1
            log_ok(f"Accepted  ({latency_ms/1000:.1f}s)  |  Raw: {len(raw_tmpl)}c → Polished: {len(polished)}c")
        else:
            log_err(f"Rejected: {failure_reason}  ({latency_ms/1000:.1f}s)")
            
        rows.append({
            "condition": display_name,
            "latency_ms": round(latency_ms, 1),
            "accepted": is_valid,
            "failure_reason": failure_reason,
            "raw_length": len(raw_tmpl),
            "polished_length": len(polished)
        })
        
    print()
    total = len(latencies)
    if total == 0:
        return {}
        
    acc_pct = (accepted_count / total) * 100
    avg_lat = statistics.mean(latencies)
    
    log_info(f"LLM Polish Accepted: {accepted_count}/{total}  ({acc_pct:.1f}%)")
    log_info(f"Avg Latency        : {avg_lat:.0f} ms")
    
    summary = {
        "total_tested": total,
        "accepted_count": accepted_count,
        "acceptance_rate_pct": round(acc_pct, 2),
        "avg_latency_ms": round(avg_lat, 1),
        "per_condition": rows
    }
    log_info(f"Report → {save_report('llm_polish', rows)}")
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# =════════════════════════════════════════════════════════════════════════════


def main() -> None:
    global BASE_URL
    parser = argparse.ArgumentParser(
        description="DermLIP Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--test",
        choices=["all", "classify", "kb", "latency", "chat", "llm"],
        default="all",
        help="Which module(s) to run  (default: all).",
    )
    parser.add_argument(
        "--images",
        default=None,
        metavar="DIR",
        help=(
            "Root directory whose sub-folders are labelled condition classes.\n"
            "Example: \"C:/Users/.../TEST PROD/image data\""
        ),
    )
    parser.add_argument(
        "--session",
        default=None,
        metavar="ID",
        help="Existing session_id to use for the chat module.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="k for Top-k accuracy in the classify module  (default: 5).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of requests for the latency benchmark  (default: 10).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent workers for the latency benchmark  (default: 1).",
    )
    parser.add_argument(
        "--classify-workers",
        type=int,
        default=1,
        dest="classify_workers",
        help="Parallel threads for the classify module  (default: 1, try 4 for ~4x speedup).",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        default=False,
        dest="skip_llm",
        help="Skip the server-side LLM polish during classify eval for a ~20-40x speedup.",
    )
    parser.add_argument(
        "--url",
        default=BASE_URL,
        metavar="URL",
        help=f"DermLIP server base URL  (default: {BASE_URL}).",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434/api/generate",
        metavar="URL",
        help="Ollama direct api URL for the LLM test (default: http://localhost:11434/api/generate).",
    )
    args = parser.parse_args()

    # Allow overriding the server URL from the CLI
    BASE_URL = args.url.rstrip("/")

    # ── Health check — always first ───────────────────────────────────────
    if not run_health():
        sys.exit(1)

    # ── Resolve a single probe image (first image found in --images dir) ──
    # Used as the sample for latency + chat when --images points to a directory
    probe_image: Optional[str] = None
    if args.images:
        p = Path(args.images)
        if p.is_file():
            probe_image = str(p)
        elif p.is_dir():
            probe_image = first_image_in(str(p))
            if probe_image:
                log_info(f"Probe image (latency/chat): {Path(probe_image).name}\n")

    # ── Run requested modules ─────────────────────────────────────────────
    summaries: Dict[str, Any] = {}

    if args.test in ("classify", "all"):
        if args.images and Path(args.images).is_dir():
            summaries["classify"] = run_classify(
                args.images,
                top_k=args.top_k,
                workers=args.classify_workers,
                skip_llm=args.skip_llm,
            )
        else:
            log_warn("classify module skipped — provide --images <directory>.")

    if args.test in ("kb", "all"):
        summaries["kb"] = run_kb()

    if args.test in ("latency", "all"):
        if probe_image:
            summaries["latency"] = run_latency(probe_image, n=args.n, workers=args.workers)
        else:
            log_warn("latency module skipped — provide --images <path>.")

    if args.test in ("chat", "all"):
        summaries["chat"] = run_chat(
            session_id   = args.session,
            sample_image = probe_image,
        )

    if args.test in ("llm", "all"):
        summaries["llm"] = run_llm(
            kb_path=KB_PATH,
            n=args.n,
            url=args.ollama_url
        )

    # ── Overall summary ───────────────────────────────────────────────────
    if len(summaries) > 1:
        section("OVERALL SUMMARY")
        if "classify" in summaries:
            c = summaries["classify"]
            log_info(
                f"Classify  top-1={c.get('top1_accuracy')}%  "
                f"top-{args.top_k}={c.get(f'top{args.top_k}_accuracy')}%  "
                f"safety_rej={c.get('safety_reject_pct')}%"
            )
        if "kb" in summaries:
            k = summaries["kb"]
            log_info(
                f"KB        coverage={k.get('coverage_pct')}%  "
                f"({k.get('covered_count')}/{k.get('total_conditions')} conditions)"
            )
        if "latency" in summaries:
            lt = summaries["latency"].get("classify", {})
            log_info(f"Latency   classify p50={lt.get('p50_ms')} ms")
        if "chat" in summaries:
            ch = summaries["chat"]
            log_info(
                f"Chat      ok={ch.get('ok_count')}/{ch.get('total_questions')}  "
                f"loops={ch.get('loop_count')}  "
                f"avg_len={ch.get('avg_reply_length')} chars"
            )
        if "llm" in summaries:
            l = summaries["llm"]
            log_info(
                f"LLM       accepted={l.get('accepted_count')}/{l.get('total_tested')}  "
                f"rate={l.get('acceptance_rate_pct')}%  "
                f"avg_lat={l.get('avg_latency_ms')}ms"
            )

    # Save combined report if more than one module ran
    if summaries:
        path = save_report("full_eval", summaries)
        log_info(f"\nFull combined report → {path}")


if __name__ == "__main__":
    main()
