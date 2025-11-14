import json
from pathlib import Path

# Path to your dataset JSON
json_path = Path(r"D:\PROJECT\weapon_detr\data\merged_dataset\train.json")

with open(json_path, "r") as f:
    data = json.load(f)

# Fix category IDs: map 1→0, 2→1
id_map = {1: 0, 2: 1}

# Fix categories
if "categories" in data:
    for cat in data["categories"]:
        if cat["id"] in id_map:
            cat["id"] = id_map[cat["id"]]
    print("✅ Fixed categories.")

# Fix annotations
for ann in data["annotations"]:
    old_id = ann["category_id"]
    ann["category_id"] = id_map.get(old_id, old_id)
print("✅ Fixed annotations category_ids.")

# Save fixed JSON
fixed_path = json_path.with_name("train_fixed.json")
with open(fixed_path, "w") as f:
    json.dump(data, f, indent=2)
print(f"💾 Saved fixed file to: {fixed_path}")
