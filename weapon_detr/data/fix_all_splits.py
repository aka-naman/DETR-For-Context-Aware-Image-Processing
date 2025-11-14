# fix_all_splits.py
import json
from pathlib import Path

DATA_DIR = Path(r"D:\PROJECT\weapon_detr\data\merged_dataset")
MAP = {1: 0, 2: 1}   # map original ids -> 0-based ids

def fix_file(path: Path, out_name: str):
    print(f"Processing {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # fix categories if present
    if "categories" in data:
        cats = data["categories"]
        new_cats = []
        used = set()
        for c in cats:
            new_id = MAP.get(c.get("id", c), c.get("id", c))
            used.add(new_id)
            new_cats.append({"id": new_id, "name": c.get("name", str(new_id))})
        # ensure sorted and unique categories
        new_cats_sorted = sorted({c['id']:c for c in new_cats}.values(), key=lambda x: x["id"])
        data["categories"] = new_cats_sorted
        print("  Fixed categories ->", data["categories"])

    # fix annotations
    if "annotations" in data:
        for ann in data["annotations"]:
            old = ann.get("category_id")
            ann["category_id"] = MAP.get(old, old)
        print("  Fixed annotation category_ids.")

    out_path = path.with_name(out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("  Saved:", out_path)

# Files to fix
files = [
    ("train.json", "train_fixed.json"),
    ("val.json",   "val_fixed.json"),
    ("annotations.json", "annotations_fixed.json")  # optional
]

for src, dst in files:
    p = DATA_DIR / src
    if p.exists():
        fix_file(p, dst)
    else:
        print(f"Skipping {p} (not found)")
