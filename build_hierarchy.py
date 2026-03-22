import json
import os
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# 16 focused categories (was 11).
# Rules:
#   • No category should exceed ~25 conditions.
#   • Categories are ordered from most-specific to least-specific so that
#     the first keyword match wins the right bucket.
#   • 'Other General Dermatosis' is the final catch-all for anything unmatched.
# ─────────────────────────────────────────────────────────────────────────────

# Step 1: Explicit direct overrides for conditions that keyword matching
# would misclassify (e.g. because their name contains misleading substrings).
DIRECT_OVERRIDES = {
    # Pigmentary – keywords like 'pigment' appear in many names
    'acanthosis nigricans':                    'Pigmentary Disorders',
    'ochronosis exogenous':                    'Pigmentary Disorders',
    'drug pigmentation':                       'Pigmentary Disorders',
    'post-inflammatory hyperpigmentation pih': 'Pigmentary Disorders',
    'urticaria pigmentosa mastocytosis':        'Pigmentary Disorders',
    'incontinentia pigmenti':                  'Pigmentary Disorders',
    'incontinentia pigmenti stage iii':        'Pigmentary Disorders',
    'xeroderma pigmentosum':                   'Pigmentary Disorders',
    'albinism':                                'Pigmentary Disorders',
    # Bacterial – 'leprosy' contains no simple bacterial keyword
    'leprosy tuberculoid':                     'Bacterial Infections',
    'leprosy lepromatous':                     'Bacterial Infections',
    'buruli ulcer':                            'Bacterial Infections',
    'tropical ulcer':                          'Bacterial Infections',
    'erythrasma':                              'Bacterial Infections',
    'erythema nodosum':                        'Bacterial Infections',
    # Parasitic / tropical
    'onchocerciasis':                          'Parasitic & Tropical',
    'loiasis':                                 'Parasitic & Tropical',
    'lymphatic filariasis':                    'Parasitic & Tropical',
    'cutaneous larva migrans':                 'Parasitic & Tropical',
    'cutaneous leishmaniasis':                 'Parasitic & Tropical',
    'myiasis':                                 'Parasitic & Tropical',
    'tungiasis jigger flea':                   'Parasitic & Tropical',
    # Viral – hand foot mouth has no viral keyword
    'hand foot mouth disease':                 'Viral Infections',
    'molluscum giant hiv':                     'Viral Infections',
    # Psoriasis & lichenoid
    'psoriatic arthritis skin':                'Psoriasis & Lichenoid',
    'pityriasis rosea':                        'Psoriasis & Lichenoid',
    'pityriasis rubra pilaris prp':            'Psoriasis & Lichenoid',
    'pityriasis lichenoides chronica plc':     'Psoriasis & Lichenoid',
    'lichen simplex chronicus':                'Psoriasis & Lichenoid',
    'neurodermatitis lichen simplex':          'Psoriasis & Lichenoid',
    # Blistering
    'hailey hailey disease':                   'Blistering Disorders',
    'epidermolysis bullosa':                   'Blistering Disorders',
    'stevens johnson syndrome ten':            'Blistering Disorders',
    'erythema multiforme':                     'Blistering Disorders',
    # Autoimmune & connective
    'dermatomyositis':                         'Autoimmune & Connective',
    'vasculitis leukocytoclastic':             'Autoimmune & Connective',
    'sarcoidosis skin':                        'Autoimmune & Connective',
    'sarcoidosis':                             'Autoimmune & Connective',
    'behcets disease':                         'Autoimmune & Connective',
    'pyoderma gangrenosum':                    'Autoimmune & Connective',
    'neutrophilic dermatoses sweet syndrome':  'Autoimmune & Connective',
    'erythema annulare centrifugum':           'Autoimmune & Connective',
    'erythema elevatum diutinum':              'Autoimmune & Connective',
    'lyme disease erythema migrans':           'Autoimmune & Connective',
    'morphea localized':                       'Autoimmune & Connective',
    # Urticaria & drug reactions
    'drug morbilliform eruption':              'Urticaria & Drug Reactions',
    'fixed drug eruption':                     'Urticaria & Drug Reactions',
    'papular urticaria':                       'Urticaria & Drug Reactions',
    # Scarring & structural
    'keloid':                                  'Scarring & Structural',
    'hypertrophic scar':                       'Scarring & Structural',
    'atrophic scar':                           'Scarring & Structural',
    'striae distensae rubrae':                 'Scarring & Structural',
    'striae distensae albae':                  'Scarring & Structural',
    'necrobiosis lipoidica diabeticorum':      'Scarring & Structural',
    'calcinosis cutis':                        'Scarring & Structural',
    'scleromyxedema':                          'Scarring & Structural',
    'mucinosis focal':                         'Scarring & Structural',
    'porphyria cutanea tarda':                 'Scarring & Structural',
    'aplasia cutis':                           'Scarring & Structural',
    'papillomatosis confluentes reticulate':   'Scarring & Structural',
    # Genodermatoses → Scarring & Structural (closest visual bucket)
    'ehlers danlos syndrome':                  'Scarring & Structural',
    'tuberous sclerosis':                      'Scarring & Structural',
    'dariers disease':                         'Scarring & Structural',
    'ichthyosis vulgaris':                     'Scarring & Structural',
    'ichthyosis lamellar':                     'Scarring & Structural',
    # Vascular
    'port wine stain':                         'Vascular Conditions',
    'telangiectases':                          'Vascular Conditions',
    'livedo reticularis':                      'Vascular Conditions',
    'livedo racemosa':                         'Vascular Conditions',
    'lymphangioma':                            'Vascular Conditions',
    'lymphangioma circumscriptum':             'Vascular Conditions',
    # Benign tumors / growths
    'skin tag acrochordon':                    'Benign Tumors & Nevi',
    'lipoma':                                  'Benign Tumors & Nevi',
    'syringoma':                               'Benign Tumors & Nevi',
    'fordyce spots':                           'Benign Tumors & Nevi',
    'xanthomas xanthelasma':                   'Benign Tumors & Nevi',
    'langerhans cell histiocytosis':           'Benign Tumors & Nevi',
    'juvenile xanthogranuloma':                'Benign Tumors & Nevi',
    'mucous cyst digital myxoid':              'Benign Tumors & Nevi',
    'halo nevus':                              'Benign Tumors & Nevi',
    'halo nevus sutton':                       'Benign Tumors & Nevi',
}

# Step 2: Keyword-based fallback categories (ordered most-specific first).
CATEGORIES = {
    'Fungal Infections': [
        'tinea', 'candidiasis', 'oral thrush', 'mycetoma', 'chromoblastomycosis',
        'sporotrichosis', 'onychomycosis', 'pityriasis versicolor', 'fungal',
    ],
    'Bacterial Infections': [
        'impetigo', 'cellulitis', 'erysipelas', 'folliculitis', 'furuncle', 'boil',
        'carbuncle', 'pseudofolliculitis', 'bacterial',
    ],
    'Viral Infections': [
        'herpes', 'wart', 'verruca', 'molluscum', 'varicella', 'chickenpox', 'zoster',
        'shingles', 'measles', 'hiv', 'viral', 'hpv',
    ],
    'Parasitic & Bug Bites': [
        'scabies', 'pediculosis', 'nematode', 'strongyloides', 'parasite', 'bug bite', 'insect bite'
    ],
    'Parasitic & Tropical': [
        'scabies', 'pediculosis', 'nematode', 'strongyloides', 'parasite',
    ],
    'Acne & Follicular': [
        'acne', 'comedonal', 'comedone', 'hidradenitis', 'milia', 'miliaria',
        'rosacea', 'rhinophyma', 'perioral',
    ],
    'Eczema & Dermatitis': [
        'atopic dermatitis', 'contact dermatitis', 'seborrheic dermatitis',
        'nummular eczema', 'dyshidrotic', 'photodermatitis', 'stasis dermatitis',
        'factitial', 'neurotic', 'prurigo', 'acrodermatitis', 'eczema', 'dermatitis',
    ],
    'Psoriasis & Lichenoid': [
        'psoriasis', 'lichen planus', 'lichen nitidus', 'lichen sclerosus',
        'pityriasis', 'lichen simplex',
    ],
    'Blistering Disorders': [
        'bullous pemphigoid', 'pemphigus', 'epidermolysis', 'bullous',
    ],
    'Autoimmune & Connective': [
        'lupus', 'dermatomyositis', 'scleroderma', 'alopecia', 'vitiligo',
        'vasculitis', 'sarcoid', 'morphea', 'behcet',
    ],
    'Urticaria & Drug Reactions': [
        'urticaria', 'angioedema', 'drug eruption', 'morbilliform',
    ],
    'Pigmentary Disorders': [
        'melasma', 'chloasma', 'acanthosis', 'ochronosis', 'albinism',
        'pigment', 'mastocytosis',
    ],
    'Skin Cancers': [
        'melanoma', 'carcinoma', 'kaposi', 'merkel', 'mycosis fungoides',
        'malignant', 'sarcoma', 'cutaneous lymphoma',
    ],
    'Benign Tumors & Nevi': [
        'nevus', 'nevi', 'seborrheic keratosis', 'dermatofibroma', 'lipoma',
        'epidermoid cyst', 'pilar cyst', 'pilomatricoma', 'granuloma',
        'neurofibromatosis', 'xanthoma', 'xanthelasma', 'syringoma', 'fordyce',
        'skin tag', 'acrochordon', 'keratosis pilaris', 'porokeratosis',
        'epidermal nevus', 'lentigo', 'tumor',
    ],
    'Vascular Conditions': [
        'hemangioma', 'angioma', 'telangiectasi', 'livedo', 'port wine',
        'lymphangioma', 'purpura', 'stasis',
    ],
    'Scarring & Structural': [
        'keloid', 'scar', 'striae', 'necrobiosis', 'calcinosis', 'scleromyxedema',
        'mucinosis', 'porphyria', 'ichthyosis', 'ehlers', 'darier', 'aplasia',
        'tuberous sclerosis', 'papillomatosis',
    ],
    # Catch-all – never matched by the blocks above
    'Other General Dermatosis': [''],  # placeholder, never actually tested
}


def normalise_key(raw_name: str) -> str:
    """Mirror the key-normalisation used in app.py."""
    import re
    return re.sub(r'[^a-z0-9_]', '', raw_name.replace(' ', '_').replace('-', '_').lower())


def main():
    cond_path = 'conditions.txt'
    if not os.path.exists(cond_path):
        print('Error: conditions.txt not found.')
        return

    mapping = {}

    with open(cond_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Extract the raw display name (part before ' — ')
            raw_cond = line.split(' — ')[0].strip()
            normalized_key = normalise_key(raw_cond)
            raw_lower = raw_cond.lower()

            # ── Priority 1: explicit override ────────────────────────────────
            assigned_category = None
            for override_name, cat in DIRECT_OVERRIDES.items():
                if override_name == raw_lower:
                    assigned_category = cat
                    break

            # ── Priority 2: keyword fallback ─────────────────────────────────
            if assigned_category is None:
                for cat, keywords in CATEGORIES.items():
                    if cat == 'Other General Dermatosis':
                        continue
                    if any(kw in raw_lower for kw in keywords if kw):
                        assigned_category = cat
                        break

            # ── Priority 3: catch-all ────────────────────────────────────────
            if assigned_category is None:
                assigned_category = 'Other General Dermatosis'

            mapping[normalized_key] = assigned_category

    # ── Write output ─────────────────────────────────────────────────────────
    with open('hierarchy_map.json', 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=4)

    # ── Print category distribution ──────────────────────────────────────────
    dist = Counter(mapping.values())
    print(f'\nGenerated hierarchy_map.json  ({len(mapping)} conditions, {len(dist)} categories)\n')
    for cat, count in sorted(dist.items(), key=lambda x: -x[1]):
        bar = '█' * count
        print(f'  {cat:<35} {count:>3}  {bar}')
    print()


if __name__ == '__main__':
    main()
