# `hierarchy_map.json` — Code Explanation

`hierarchy_map.json` is the **super-category bucketing file** used by the Phase 2 Hierarchical Classification pipeline. It maps every condition key in the system to one of 11 broad disease super-categories, enabling a 2-pass coarse-to-fine classification strategy inside `app.py`.

---

## How It Is Generated

This file is **auto-generated** by `build_hierarchy.py` — you never edit it by hand.

```bash
python build_hierarchy.py
```

The script parses `conditions.txt`, extracts each condition key, and uses keyword matching to assign it to the most appropriate super-category. Run this script again any time new conditions are added to `conditions.txt`.

---

## File Format

A flat JSON object mapping `condition_key → super_category_string`:

```json
{
    "tinea_corporis":       "Fungal Infections",
    "acne_inflammatory":    "Acne & Follicular",
    "melanoma_superficial": "Skin Cancers & Benign Tumors",
    "vitiligo":             "Pigmentary Disorders",
    ...
}
```

---

## The 11 Super-Categories

| Super-Category | Approx. Count |
|---|---|
| Other General Dermatosis | 90 |
| Skin Cancers & Benign Tumors | 52 |
| Autoimmune & Inflammatory | 46 |
| Eczema & Dermatitis | 17 |
| Acne & Follicular | 16 |
| Fungal Infections | 12 |
| Parasitic & Bug Bites | 10 |
| Pigmentary Disorders | 10 |
| Bacterial Infections | 9 |
| Viral Infections | 9 |
| Vascular Conditions | 6 |

---

## How It Is Used in app.py

At server startup, `app.py` loads this map and builds a **reverse lookup**:

```python
HIERARCHY_MAP    = _load_hierarchy_map("/app/hierarchy_map.json")
CATEGORY_BUCKETS = _build_category_buckets(SKIN_CONDITIONS, HIERARCHY_MAP)
```

`CATEGORY_BUCKETS` groups all full condition strings by their super-category.

During each `/classify` call:

1. **Pass 1 (Soft Category Guess)**: The image is compared against the 11 super-category labels to identify the likely "disease family".
2. **Pass 2 (Soft Re-ranking with Bonus)**:
    - **All 277 conditions** are always searched (no more hard gating).
    - Conditions belonging to the predicted category receive a **+1.5 logit bonus**.
    - This allows the AI to correctly "group" visually similar diseases while preventing a wrong category guess from hiding the correct answer entirely.

This research-backed approach (based on ICCV 2023 WaffleCLIP) maximizes both precision and recall by providing a "gentle nudge" instead of a restrictive filter.

---

## Fallback Behavior

If the file is missing or a condition key is not found in the map, `app.py` falls back safely to searching the full vocabulary. You will see a warning in the container logs:

```
WARNING: Could not load hierarchy map: [error]. Falling back to flat search.
```

---

## When to Regenerate

- When new conditions are added to `conditions.txt`
- When you want to re-categorize existing conditions

After regenerating, restart the container (`docker compose down && docker compose up`) — **no rebuild needed**.
