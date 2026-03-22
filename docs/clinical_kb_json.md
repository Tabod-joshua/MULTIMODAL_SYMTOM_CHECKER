# `clinical_kb.json` — Complete Line-by-Line Code Explanation

`clinical_kb.json` is the **single source of medical truth** for the entire DermLIP system. It is a structured database containing deterministic, pre-verified clinical information for every skin condition the server can identify. When the CLIP model makes a prediction, the server immediately looks up the matching entry in this file to retrieve medically accurate descriptions, causes, and management guidance.

The philosophy: **no LLM should invent a medical fact from scratch**. All core medical information comes from this database. The LLM's only allowed role is to rephrase that information more naturally.

> **Phase 2 Status:** The KB is now fully complete — **278 entries**, **0 unknown placeholders**. All 111 previously missing image-dataset conditions have been injected with medically accurate, individually verified profiles covering risk level, causes, management, and emergency symptoms.

See also: [`hierarchy_map_json.md`](./hierarchy_map_json.md) — the companion file that maps all 277 conditions to 11 super-categories for the Hierarchical 2-Pass Classification pipeline.

---

## File Format Overview

The file is a single large JSON object where:
- **Keys**: Normalized condition identifiers (lowercase, underscores, no special chars) — e.g. `"acne_inflammatory_papulopustular"`.
- **Values**: Entry objects with exactly defined fields.

The server reads this file via `_load_clinical_kb()` in `app.py` during startup. The key normalization in the lookup code (`re.sub(r'[^a-z0-9_]', '', raw_name.replace(" ", "_").lower())`) must match the keys in this file exactly. If there is a mismatch, the condition falls through to `generic_fallback`.

---

## Special Section Headers

```json
"__SECTION_FUNGAL INFECTIONS__": "================ FUNGAL INFECTIONS ================"
```
Lines like this (keys starting with `__SECTION_`) are organizational markers. The `_load_clinical_kb()` startup validator skips entries whose key starts with `_`, so these entries are never validated or used as condition data. They exist purely so the JSON file is organized and human-readable in a text editor.

---

## A Complete Entry — Every Field Explained

Using `acne_inflammatory_papulopustular` as the reference example:

### `"condition_id"` (string)
```json
"condition_id": "acne_inflammatory_papulopustular"
```
**What it is**: Mirrors the dictionary key exactly. A self-referencing identifier.
**Why it exists**: Allows the entry to be self-describing when extracted from the dict. If you get this entry as an isolated object (e.g. from a database query), you don't lose context on what condition it represents.

### `"display_name"` (string)
```json
"display_name": "Acne Inflammatory Papulopustular"
```
**What it is**: The human-readable, properly-cased name shown to users in the API response (`top_prediction`) and in the UI.
**Critical role**: After the KB lookup, `app.py` executes `display_name = kb_entry.get("display_name", display_name)`. This syncs the display name from the KB (the authoritative source) overriding the `clean_display_name()` computed version. This ensures the displayed name always matches exactly what the KB says, which is also used for diagnosis drift validation in the LLM.

### `"category"` (string)
```json
"category": "inflammatory"
```
**What it is**: Broad condition family (e.g. `"fungal infections"`, `"bacterial infections"`, `"inflammatory"`, `"benign neoplasms"`).
**Why it exists**: Enables future filtering and grouping in analytics. Currently logged but not yet used in routing logic.

### `"subcategory"` (string)
```json
"subcategory": "acne"
```
**What it is**: More specific sub-grouping within the category.
**Why it exists**: Clinical reporting and future grouping (e.g. grouping all acne variants together in a report).

### `"risk_level"` (string)
```json
"risk_level": "low"
```
**What it is**: The clinical severity tier. Used by `app.py` to return `clinical_severity` in the response.
**Values**: `"low"` (self-limiting, minor), `"moderate"` (requires medical attention), `"high"` (urgent care), `"critical"` (emergency), `"unknown"` (generic fallback).
**How it's used**: The `evaluate.py` script reads this from the response and records it in the per-image CSV. In future UI work, this field could be used to color-code severity on a patient dashboard.

### `"severity_weight"` (integer)
```json
"severity_weight": 1
```
**What it is**: A numeric version of risk level (1=low, 2=moderate, 3=high, 4=critical) for sorting and computation.
**Why it exists**: Enables arithmetic on severity. For example, averaging severity weights across a session to compute an overall risk score, or sorting a differential diagnosis list by urgency.

### `"allow_self_care"` (boolean)
```json
"allow_self_care": true
```
**What it is**: A clinical safety flag. If `true`, the LLM is allowed to give general skincare advice in the `/chat` endpoint.
**Critical use in `/chat`**: The `self_care_directive` variable in `app.py`:
```python
self_care_directive = (
    "You may suggest general skincare advice."
    if session.get('allow_self_care', False)
    else "Advise the patient to see a doctor for treatment options."
)
```
For conditions like `"acne"` (allow_self_care=true), the chatbot can say "try a benzoyl peroxide cleanser." For conditions like `"tinea_corporis"` (allow_self_care=false), it can only tell the patient to see a doctor, because antifungal treatment decisions require clinical judgment.

### `"review_status"` (string)
```json
"review_status": "clinically_verified"
```
**What it is**: Audit trail status. Values: `"clinically_verified"` or `"draft"`.
**Why it exists**: When a new entry is added quickly, it should be marked as `"draft"` until a qualified clinician has reviewed and confirmed the information. This supports a future compliance workflow where `"draft"` entries could trigger a warning or lower confidence score in the response.

### `"description"` (string)
```json
"description": "A form of acne characterized by red inflamed papules and pustules..."
```
**What it is**: A single-paragraph patient-facing clinical description of the condition.
**How it's used**: Injected into `build_raw_template()` as the `Description:` section of the deterministic output. This is the most important field — it is the core medical explanation the user receives. It must be accurate, factual, and non-alarmist.

### `"common_causes"` (string)
```json
"common_causes": "Excess oil production, hormonal changes..."
```
**What it is**: A concise list of the primary etiological factors.
**Why single paragraph**: The template builder simply inserts this as a `Common causes:` paragraph. A list format would need extra handling. Using a comma-separated prose sentence keeps the output clean.

### `"general_management"` (string)
```json
"general_management": "Non-comedogenic skincare, benzoyl peroxide..."
```
**What it is**: General, safe, and non-specific advice about management.
**Critical constraint**: This must NOT be specific treatment prescriptions (e.g. "Take 100mg doxycycline twice daily"). It should describe the general *approach* to care (e.g. "Antifungal medication as directed by a doctor"). The disclaimer and `allow_self_care` flag together ensure clinical safety for the specific depth of advice given.

### `"when_to_seek_medical_care"` (string)
```json
"when_to_seek_medical_care": "Severe pain, spreading redness, fever..."
```
**What it is**: A sentence describing the tipping point when the patient should stop home management and see a doctor non-urgently.
**How it's used**: Appears in `build_raw_template()` as the `Seek medical care if:` section at the end of the deterministic template.

### `"emergency_symptoms"` (list of strings)
```json
"emergency_symptoms": [
    "The rash or swelling is spreading rapidly.",
    "You develop a high fever.",
    "The area is severely painful..."
]
```
**What it is**: A list of specific red-flag symptoms warranting emergency care.
**Critical role**: This list is returned as `seek_emergency_care_immediately_if` in the API response **independently** — it is not processed by the LLM and cannot be altered by the polish step. This is the hardest safety boundary in the entire system. Whatever the LLM does, the emergency checklist arrives verbatim from the KB.
**Why a list, not a string**: Allows the frontend to render each item as a bullet point directly, making it easy to scan in an emergency situation.

### `"confidence_guidance"` (string)
```json
"confidence_guidance": "This prediction may not be fully accurate."
```
**What it is**: Entry-specific note about the reliability of this prediction.
**Why it varies**: For common, visually distinctive conditions (like typical acne), the model is likely accurate. For rare or visually ambiguous conditions, this field should say something like "This is a rare condition with several visual mimics. A biopsy may be required for confirmation."

---

## The `generic_fallback` Entry

```json
"generic_fallback": {
    "condition_id": "generic_fallback",
    "display_name": "Unknown Skin Condition",
    ...
    "allow_self_care": false,
    ...
}
```
**What it is**: The default entry used when the CLIP model predicts a condition that has no matching entry in the KB.
**Why `allow_self_care: false`**: If we don't know what the condition is, we cannot safely allow self-care advice. The user is always directed to see a doctor.
**Emergency symptoms**: Generic but comprehensive — covers the most common reasons any unknown skin condition should trigger an emergency visit (spreading redness, fever, breathing difficulty).
**Design principle**: Fail safe, not silently. If the KB has a gap, the response degrades gracefully to generic but medically responsible advice, never to an empty response or an exception.
