# `evaluate.py` — Comprehensive Code Explanation

This document provides a detailed, line-by-line breakdown of the DermLIP Evaluation Suite. It is the test harness that measures the server's performance across four distinct modules. Unlike the server (`app.py`), this script runs from your local Windows machine and connects to the server via HTTP.

---

## 1. Module Docstring (Lines 1–44)

```python
"""
DermLIP Server Evaluation Suite
================================
USAGE:
    python evaluate.py --images "C:/Users/.../TEST PROD/image data"
...
"""
```
The triple-quoted string at the top of the file is a **module docstring**. Python uses it as the help text printed if you run `python evaluate.py --help`. It documents: usage commands, how to structure image folders, pip requirements, and what reports are generated. It is referenced later in `parser.epilog = __doc__` in `main()`.

---

## 2. Imports (Lines 46–64)

### Standard Library Imports (Lines 46–57)
```python
import argparse, csv, json, os, re, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
```
- **`argparse`**: Parses command-line arguments (`--images`, `--skip-llm`, etc.)
- **`csv`**: Writes per-image result rows to `.csv` spreadsheet files.
- **`json`**: Reads `clinical_kb.json` and writes JSON reports.
- **`os`**: Used for `os.makedirs()` to create the `eval_reports/` directory.
- **`re`**: Regular expressions for loop detection (`is_looping`) and label normalization.
- **`statistics`**: For `statistics.mean()` — calculates average latency and average reply length.
- **`sys`**: For `sys.exit(1)` — force-exits with an error code if the health check fails.
- **`time`**: For `time.perf_counter()` — high-precision timer for latency measurement.
- **`ThreadPoolExecutor, as_completed`**: For parallel image processing in `run_classify()`. `as_completed` yields futures as they finish, regardless of order.
- **`datetime`**: For timestamping report filenames (e.g. `classify_2026-03-14_15-30-00.json`).
- **`Path`**: `pathlib.Path` — used for clean cross-platform file path handling (important because the user runs this on Windows while referring to paths).
- **`Optional, Dict, List, Tuple, Any`**: Type hints for function signatures, making the code self-documenting.

### Third-Party Import with Error Handling (Lines 59–64)
```python
try:
    import requests
except ImportError:
    print("  [ERROR] 'requests' is not installed. Run: pip install requests")
    sys.exit(1)
```
Wraps the `requests` import in a `try/except`. If `requests` isn't installed, instead of a confusing Python traceback, the user gets a clear fix instruction and the program exits cleanly. `sys.exit(1)` returns exit code 1 (convention for error), which CI/CD systems use to detect failures.

---

## 3. Configuration Constants (Lines 67–95)

### Lines 71–75: Paths and Timeouts
```python
BASE_URL        = "http://localhost:8000"
KB_PATH         = "clinical_kb.json"
CONDITIONS_PATH = "conditions.txt"
REPORT_DIR      = "eval_reports"
REQUEST_TIMEOUT = 180
```
- **`BASE_URL`**: The default server URL. Can be overridden with `--url` flag if the server is remote.
- **`KB_PATH / CONDITIONS_PATH`**: Relative paths to the data files on disk. These are read locally (not via the API) for offline analysis in Module 2 (KB Coverage).
- **`REPORT_DIR`**: All JSON and CSV reports are saved here. Created automatically if it doesn't exist.
- **`REQUEST_TIMEOUT = 180`**: Maximum seconds to wait for a server response. Set to 180 to accommodate the LLM polish step in `app.py` which can take 30–120s.

### Lines 77–88: Chat Test Questions
```python
CHAT_QUESTIONS: List[str] = [
    "Is this condition contagious?",
    "What causes this condition?",
    ...
]
```
Eight standardized questions for Module 4 (Chat Quality). Standardizing these questions makes the results reproducible — running the evaluation twice with the same questions gives comparable results. They cover the most common patient concerns: contagiousness, causes, home treatment, lifestyle, children, diagnosis certainty, recovery speed, and emergency symptoms.

### Lines 90–95: Fuzzy Match Stop Words
```python
FUZZY_STOP_WORDS = {"skin", "infection", "fungal", "viral", "bacterial", "rash",
                    "disease", "condition", "disorder", "lesion", "syndrome"}
```
Used in `folder_matches_prediction()`. These are medically generic words that appear in both the folder names AND in many different predicted condition names. If these were included in the matching logic, they would cause false-positive matches (e.g. the folder `tinea_corporis` has "infection," and the prediction `psoriasis skin condition` also has "skin" — wrongly marking it as correct). By removing them, only truly discriminating words are used for the comparison.

---

## 4. Terminal UI Helpers (Lines 98–127)

### `_ansi_supported()` (Lines 102–107)
```python
def _ansi_supported() -> bool:
    if sys.platform == "win32":
        return bool(os.environ.get("WT_SESSION") or "ANSICON" in os.environ)
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
```
Detects if the current terminal supports ANSI color codes for colored output. `WT_SESSION` is set by Windows Terminal. `ANSICON` is set by the ANSICON driver. On non-Windows (Linux, macOS), it checks if stdout is an interactive terminal (not redirected to a file).

### Color Codes (Lines 109–115)
```python
if _ansi_supported():
    SYM_OK   = "\033[92m✔\033[0m"
    SYM_WARN = "\033[93m⚠\033[0m"
    SYM_ERR  = "\033[91m✖\033[0m"
    SYM_INFO = "\033[94m•\033[0m"
else:
    SYM_OK, SYM_WARN, SYM_ERR, SYM_INFO = "[OK]", "[WARN]", "[ERR]", "[INFO]"
```
ANSI codes: `\033[92m` = bright green, `\033[93m` = yellow, `\033[91m` = bright red, `\033[94m` = blue. `\033[0m` resets color. If ANSI isn't supported, falls back to plain-text tags.

### `section()`, `log_ok()`, etc. (Lines 118–127)
Wrapper functions for `print()` that prepend the correct symbol. `section()` prints a divider line and title in bold to visually separate the 4 modules in the terminal output.

---

## 5. API Helper Functions (Lines 130–170)

### `api_classify(image_path, skip_llm=False)` (Lines 134–154)
```python
def api_classify(image_path: str, skip_llm: bool = False) -> Optional[Dict[str, Any]]:
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
        log_err(f"classify failed [{Path(image_path).name}]: {exc}")
        return None
```
- Opens the image file in binary read mode (`"rb"`) — required for HTTP multipart uploads.
- Sends as `multipart/form-data` using `files={"file": fh}` which is what FastAPI's `UploadFile` expects.
- `params={"skip_llm": "true"}` appends `?skip_llm=true` to the URL query string when in fast evaluation mode, telling the server to skip the 30–120s LLM step.
- **`resp.raise_for_status()`**: Throws an exception for 4xx/5xx HTTP error codes. Without this, a `500 Internal Server Error` would silently return with an empty body.
- Returns `None` on any error (network, timeout, HTTP error). The `None` return is checked by callers — if `None`, the image is logged as `api_error` and processing moves on. This prevents a single crashed image from aborting the entire 3,500-image run.

### `api_chat(session_id, message)` (Lines 154–170)
Similar structure. POSTs JSON body (`{"session_id": ..., "message": ...}`) to `/chat`. Used by Module 4 to automate the 8 follow-up questions in the chat quality test.

---

## 6. Utility / Math Functions (Lines 173–299)

### `normalise(text)` (Lines 176–179)
```python
def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())
```
Strips everything except lowercase letters and numbers. Used inside `folder_matches_prediction()` to prepare both the folder label and the prediction for word-level comparison.

### `folder_matches_prediction(folder_name, predicted)` (Lines 181–207)
The core fuzzy matching function for classification accuracy. Steps:
1. Normalizes both inputs (lowercase, special chars → spaces).
2. Splits into word sets.
3. **Removes stop words** from the folder label's word set.
4. **Checks word overlap**: if ≥ 50% of the meaningful folder-label words appear in the prediction, it's a match.
5. Falls back to original match check if stop word removal leaves nothing.

**Why 50% not 100%?** A condition like `"acne vulgaris comedonal"` in the folder should match if the model predicts `"Acne Vulgaris"` (which is missing `"comedonal"` but matches the core 2 words). 100% would be too strict.

### `normalise_to_kb_key(condition_str)` (Lines 209–219)
Mirrors the exact normalization logic from `app.py`:
1. Strip text after ` — ` (the visual description part).
2. Replace spaces with `_`.
3. Remove all non-alphanumeric characters.
4. Lowercase.

This function is used in Module 2 (KB Coverage) to check whether each line in `conditions.txt` has a corresponding KB entry, entirely offline without making API calls.

### `is_looping(text)` (Lines 222–247)
Three-layer anti-stutter detection used in Module 4:
1. **Regex substring repetition**: `(.{2,20})\1{3,}` — any 2–20 char pattern repeated 4+ times consecutively. Catches "the the the the" and "ha ha ha ha".
2. **Vocabulary collapse ratio**: If unique words / total words < 25%, the model has entered a degenerate repetition state. A healthy reply should have variety.
3. **Sentence deduplication**: If > 40% of sentences are identical (after normalization), it's looping at the sentence level.

### `percentile(data, pct)` (Lines 250–264)
Pure Python percentile calculation. Sorts the numeric list, calculates the exact index at the given percentage, and linearly interpolates between adjacent values. Used for p50 (median), p95, p99 latency calculation in Module 3 without requiring numpy.

### `save_report(name, data)` (Lines 267–290)
Creates `eval_reports/` if it doesn't exist. Generates a timestamp-based filename (`classify_2026-03-14_15-30.json`). Writes JSON. If data is a flat list of dicts (not nested), also writes a `.csv` version using `csv.DictWriter` — useful for opening in Excel/Sheets.

### `first_image_in(root)` (Lines 293–299)
Recursively walks the directory tree looking for the first file with a valid image extension (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`). Returns its full path as a string. Used at startup to identify the "probe image" for Modules 3 and 4.

---

## 7. Module 0: Server Health (Lines 306–324)

```python
def run_health() -> bool:
    resp = requests.get(f"{BASE_URL}/health", timeout=10)
    data = resp.json()
    log_ok(f"Status : {data.get('status', '?')}")
    log_ok(f"Device : {data.get('device', '?')}")
    log_ok(f"LLM    : {data.get('llm', '?')}")
    return data.get("status") == "ok"
```
Fails fast: if the server isn't running, `requests.get()` raises `ConnectionError`, the `except` block catches it, logs `[ERR] Server is not reachable`, and returns `False`. `main()` then calls `sys.exit(1)`, stopping the entire evaluation before wasting time on 3,500 image requests against a dead server.

---

## 8. Module 1: Classification Accuracy (Lines 325–507)

### Function Signature
```python
def run_classify(images_dir: str, top_k: int = 3, workers: int = 1, skip_llm: bool = False) -> Dict[str, Any]:
```
All four parameters are passed from the CLI flags: `--images`, `--top_k`, `--classify-workers`, `--skip-llm`.

### Directory Ingestion (Lines 350–384)
`Path(images_dir).iterdir()` lists all items in the directory. Only sub-directories are kept (each sub-directory is a condition class label). For each folder, `Path(folder).glob("*.[jJpP][pPnN][gG]")` (and similar patterns) finds all image files recursively. The folder NAME is the ground truth label (e.g. folder `acne_vulgaris` = the images inside are all confirmedly acne vulgaris).

### `_process_one(img_path)` Inner Function (Lines 387–427)
This function processes a single image and returns a dictionary of metrics. Being a closure, it can access `label`, `skip_llm`, `top_k` from the outer scope without arguments:
- Calls `api_classify()` and times the duration using `time.perf_counter()`.
- **KB Hit Detection**: `raw_expl` (the un-LLM'd template) is empty if the lookup hit `generic_fallback`. A non-generic entry means the model found a KB entry.
- **LLM Polish Detection**: Compares `explanation` (the LLM-processed text) body against `raw_expl`. If they're identical and a KB entry exists, the LLM was rejected and the fallback was used.
- **Tone verification**: Checks if the confidence prefix line ("This prediction has high confidence.") matches the confidence percentage tier (≥75% = high, ≥50% = moderate, else limited).
- **`folder_matches_prediction()`** is called for both top-1 and the top-k list.

### Parallel Execution (Lines 440–452)
```python
if workers > 1:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one, p): p for p in img_files}
        label_results = [fut.result() for fut in as_completed(futures)]
else:
    label_results = [_process_one(p) for p in img_files]
```
With `--classify-workers 4`, up to 4 images are sent to the server simultaneously. `ThreadPoolExecutor` manages the thread pool. `as_completed` yields futures in the order they finish (not the order they were submitted), so faster responses don't wait for slower ones.

### Aggregation (Lines 454–507)
After all images in a folder are processed, their results are aggregated into global counters (`top1_hits`, `topk_hits`, `kb_hits`, etc.). The per-folder summary line is printed to the terminal using `log_ok` (if perfect), `log_warn` (if partially correct), or `log_err` (if 0 correct). The full per-image results are accumulated in `rows` for the CSV report.

---

## 9. Module 2: KB Coverage (Lines 512–585)

```python
def run_kb() -> Dict[str, Any]:
```
Performs offline gap analysis — no API calls:
1. Reads every line from `conditions.txt` (the condition descriptors).
2. Applies `normalise_to_kb_key()` to each to get its expected KB key.
3. Loads `clinical_kb.json` directly from disk.
4. Checks each key against the KB dict.
5. Reports what percentage of conditions have KB entries and lists the uncovered ones.

This tells you how much of the dataset is being handled by specific clinical data vs. the generic fallback.

---

## 10. Module 3: Latency Benchmark (Lines 592–700)

### Threaded Stress Test (Lines 621–640)
```python
with ThreadPoolExecutor(max_workers=workers) as pool:
    futures = [pool.submit(_classify_once, sample_image) for _ in range(n)]
    for fut in as_completed(futures):
        ms, sid = fut.result()
        classify_ms.append(ms)
```
Fires `n` requests concurrently (default 10). Records both the latency and the session ID from the last response (used to test chat latency right after). Reports min, mean, p50, p95, p99 latencies and overall throughput (requests per second).

### Chat Latency (Lines 657–679)
After the classify benchmark, immediately tests chat latency by posting each of the 8 `CHAT_QUESTIONS` sequentially to `/chat` and recording their response times.

---

## 11. Module 4: Chat Quality (Lines 707–839)

### Session Bootstrapping (Lines 727–743)
```python
if session_id is None:
    resp = api_classify(str(sample_image))
    session_id = resp.get("session_id")
```
The `/chat` endpoint requires a session. If the user didn't provide `--session`, the module automatically classifies the probe image first to create a fresh session. This makes the test fully autonomous.

### Automated QA Checks Per Reply (Lines 749–809)
For each of the 8 questions, the reply is checked:
- **`is_looping(reply)`**: LLM stuck in a repetition loop.
- **`fallback`**: Detected by checking if the reply contains the server's hardcoded fallback phrase ("I'm unable to generate a response").
- **`truncated`**: Checks if the reply ends abruptly without proper sentence punctuation (`.`, `!`, `?`), suggesting the model hit its `num_predict` token limit mid-sentence.
- **`too_short`**: Reply shorter than 80 characters — likely a non-answer.
- **`has_disclaimer`**: Not currently used as a failure condition but logged for audit purposes.

### Aggregation (Lines 811–839)
Calculates rates for loops, fallbacks, truncation. Prints summary stats (OK count, loop rate, avg length, avg latency). Returns a structured summary dict and saves to `chat_quality_{timestamp}.json`.

---

## 12. Entry Point: `main()` (Lines 846–996)

### argparse Setup (Lines 847–915)
Defines 8 CLI flags:
| Flag | Default | Purpose |
|---|---|---|
| `--test` | `all` | Choose which module(s) to run |
| `--images` | None | Root directory of labeled image folders |
| `--session` | None | Reuse an existing session ID for chat test |
| `--top_k` | 3 | How many predictions count as "top-k" |
| `--n` | 10 | Number of requests for latency benchmark |
| `--workers` | 1 | Parallel threads for latency benchmark |
| `--classify-workers` | 1 | Parallel threads for classification module |
| `--skip-llm` | False | Bypass server-side LLM for ~20-40x speedup |
| `--url` | localhost:8000 | Server base URL |

### Module Dispatch (Lines 937–958)
After health check and probe image resolution, uses `args.test in ("classify", "all")` etc. to conditionally run each module. Results go into the `summaries` dict.

### Final Dashboard (Lines 960–996)
If `len(summaries) > 1` (multiple modules ran), prints a consolidated one-page summary at the very end of the terminal output showing all key metrics from all modules, then saves the full combined JSON to `full_eval_{timestamp}.json`.
