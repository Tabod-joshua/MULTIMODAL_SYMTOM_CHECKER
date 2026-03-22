import os
import json

def extract_and_add_labels():
    base_dir = r"C:\Users\HP x360 1030 G2\Desktop\TEST PROD\image data"
    
    # 1. Get all folder names (these are the raw labels)
    dirs = [d.name for d in os.scandir(base_dir) if d.is_dir()]
    
    # 2. Extract existing labels from conditions.txt
    existing_conds = []
    with open('conditions.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            raw_c = line.split(' — ')[0].strip().lower()
            existing_conds.append(raw_c)

    # 3. Find exactly which dataset labels are missing
    new_conditions_to_add = []
    for d in dirs:
        readable_name = d.replace('_', ' ').strip().lower()
        if readable_name not in existing_conds:
            # Add to conditions array with generic rich descriptor fallback
            new_conditions_to_add.append(f"{readable_name} — typical clinical presentation of {readable_name} showing affected skin areas\n")

    # 4. Append to conditions.txt
    if new_conditions_to_add:
        with open('conditions.txt', 'a', encoding='utf-8') as f:
            f.writelines(new_conditions_to_add)
            
    # 5. Add to clinical_kb.json
    with open('clinical_kb.json', 'r', encoding='utf-8') as f:
        kb = json.load(f)
        
    added_to_kb = 0
    for d in dirs:
        normalized_key = d.lower().replace(' ', '_').replace('-', '_')
        if normalized_key not in kb:
            # Add generic KB entry
            readable_name = d.replace('_', ' ').strip().title()
            kb[normalized_key] = {
                "Description": f"A dermatological condition diagnosed as {readable_name}.",
                "Key Visuals": ["affected skin area"],
                "Symptoms": ["varied clinical presentation"],
                "Triggers": ["unknown"],
                "Emergency": False
            }
            added_to_kb += 1

    with open('clinical_kb.json', 'w', encoding='utf-8') as f:
        json.dump(kb, f, indent=4)

    print(f"Extraction Complete!")
    print(f"Added {len(new_conditions_to_add)} missing labels to conditions.txt")
    print(f"Added {added_to_kb} missing labels to clinical_kb.json")

if __name__ == '__main__':
    extract_and_add_labels()
