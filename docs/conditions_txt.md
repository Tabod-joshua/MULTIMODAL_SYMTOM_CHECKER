# `conditions.txt` — Complete Line-by-Line Code Explanation

`conditions.txt` is the **taxonomy and vocabulary file** of the entire DermLIP classification system. It contains the full list of skin condition descriptors that the CLIP vision model evaluates every uploaded image against. It is the most impactful file for classification accuracy — the quality of descriptions here directly determines whether the model can distinguish between similar-looking conditions.

---

## What Is This File Doing?

When you classify an image, `app.py` loads every line from this file and creates text embeddings for each one. CLIP then compares the image embedding against all these text embeddings to find the closest match. The winning line is the model's diagnosis.

This means:
- **A more specific, visually-descriptive line = better discrimination**.
- **A vague line = the model can't tell similar conditions apart**.

---

## File Format Rules

From the header comments (lines 1-16):
```
# FORMAT RULES:
#   1. Most visually distinctive feature FIRST (CLIP weights early tokens heavily)
#   2. Then: color → texture → morphology → body location → unique differentiators
#   3. One condition per line. Lines starting with # are comments (ignored by server).
#   4. Be specific. "honey-colored crust" beats "skin problem".
#   5. Include what makes it DIFFERENT from similar conditions.
```

**Rule 1 (Visually distinctive first)**: CLIP's attention mechanism gives more weight to the beginning of text. For `"honey-colored crust oozing with papules — impetigo infection"`, CLIP locks onto `honey-colored crust` first. If you wrote `"impetigo — a bacterial infection characterized by..."`, the most generic word `impetigo` anchors CLIP's embedding rather than the distinctive visual feature.

**Rule 3 (Comments with `#`)**: The `_load_conditions()` function in `app.py` uses `if not line.strip().startswith("#")` to skip any line beginning with `#`. This allows you to add section headers, notes, and disabled conditions without needing to delete them.

**Rule 4 (Specificity)**: `"honey-colored crust"` is something a camera can see. `"skin problem"` is not. CLIP was trained on image-caption pairs from the internet, so it understands visual descriptions, colors, textures, and shapes.

---

## Structure: Sections Organized by Category

The file is divided into clearly labeled sections with `#` header comments:

```
# ════ SECTION 1: FUNGAL INFECTIONS ════
tinea_corporis — circular ring-shaped scaly red plaque with active raised border...
tinea_pedis — scaling maceration peeling skin in web spaces between toes athlete's foot...
...

# ════ SECTION 2: PARASITIC INFESTATIONS ════
scabies — intensely itchy S-shaped burrow tracks between fingers...
...
```

**Why organized like this?** For human readability only — the Python code treats every non-comment line identically. The section headers do NOT affect classification. They exist so you can quickly find and update a specific condition group.

---

## Anatomy of a Condition Line

Each condition line follows this pattern:
```
[condition-name] — [key visual anchor] [color/texture] [location] [differentiators]
```

**Example:**
```
acne vulgaris — red inflamed papules comedones pustules face back chest oily skin teenage onset
```

- `acne vulgaris`: The canonical medical name. This gets extracted and used as the display name in the API.
- `—` (em-dash): A visual separator. The `app.py` code splits on `" — "` to extract just the condition name before looking it up in `clinical_kb.json`.
- `red inflamed papules`: Primary visual anchor — what the camera sees first.
- `comedones pustules`: Secondary features that distinguish from rosacea (which has no comedones).
- `face back chest`: Location anchors — help distinguish from body acne variants.
- `oily skin teenage onset`: Contextual differentiators — CLIP has seen these associations in training data.

---

## CLIP Optimization Strategy (Why Lines Look Like They Do)

CLIP was trained on hundreds of millions of image-text pairs from the internet (stock photos, medical databases, social media images with descriptions). When it encoded text during training, it learned associations between visual features and words. The optimization strategy in the header comments is derived from this:

**"honey-colored crust"**: CLIP definitely saw captions like "baby with honey-colored crusted sores" in training data. It can map these words to a visual appearance.

**"S-shaped burrow tracks"**: CLIP has learned the geometry of curves and tracks from engineering/nature photography. It combines that with skin context to identify scabies.

**"pearly umbilicated dome papule"**: Medical photography databases use these exact clinical terms in their captions. CLIP has seen them paired with molluscum images.

This is why a medically precise, visual-first description outperforms generic medical terminology for zero-shot CLIP classification.

---

## How Adding/Editing a Line Immediately Affects the Server

1. Edit `conditions.txt` and save.
2. Run `docker compose restart` (or `docker compose up -d` if using volumes).
3. On restart, `_load_conditions()` re-reads the file. The new list becomes `SKIN_CONDITIONS`.
4. On the next `/classify` call, CLIP evaluates the image against the updated list.

No Python code changes, no Docker rebuild needed — the volume mount in `docker-compose.yml` makes it a live configuration file.

---

## Important Considerations

**Condition names must be unique**: If two lines have the same words before the `—`, the `clean_display_name()` function in `app.py` could produce identical display names, making it impossible to distinguish which KB entry to look up.

**Long lines work fine**: CLIP's tokenizer truncates at 77 tokens. Lines with extremely long descriptions beyond ~60 words will have the tail truncated. The most important features should always be placed first (Rule 1).

**KB Coverage**: Every condition in this file should have a matching entry in `clinical_kb.json`. If it doesn't, the server falls back to `generic_fallback`. This gap is what `evaluate.py`'s Module 2 (KB Coverage) measures.
