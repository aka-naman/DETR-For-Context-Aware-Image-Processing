import json
import os

knife_json = "weapon_detr/data/dataset_A/train/dataset_A.json"
gun_json = "weapon_detr/data/dataset_B/train/dataset_B.json"
output_json = r"D:\PROJECT\weapon_detr\data\merged_annotations.json"


with open(knife_json) as f:
    knife_data = json.load(f)
with open(gun_json) as f:
    gun_data = json.load(f)

merged = {
    "info": {"description": "Knife + Gun merged dataset"},
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 1, "name": "knife"},
        {"id": 2, "name": "gun"}
    ]
}

# ---- handle IDs ----
img_offset = 0
ann_offset = 0

def merge(source, cat_name):
    global img_offset, ann_offset
    imgs, anns = [], []

    cat_id = 1 if cat_name == "knife" else 2
    img_id_map = {}

    for img in source["images"]:
        new_img = img.copy()
        new_img["id"] = img["id"] + img_offset
        img_id_map[img["id"]] = new_img["id"]
        imgs.append(new_img)

    for ann in source.get("annotations", []):
        new_ann = ann.copy()
        new_ann["id"] = ann["id"] + ann_offset
        new_ann["image_id"] = img_id_map.get(ann["image_id"], ann["image_id"] + img_offset)
        new_ann["category_id"] = cat_id
        anns.append(new_ann)

    img_offset += len(imgs)
    ann_offset += len(anns)
    return imgs, anns

knife_imgs, knife_anns = merge(knife_data, "knife")
gun_imgs, gun_anns = merge(gun_data, "gun")

merged["images"].extend(knife_imgs + gun_imgs)
merged["annotations"].extend(knife_anns + gun_anns)

with open(output_json, "w") as f:
    json.dump(merged, f, indent=2)

print(f"✅ Merged dataset saved to {output_json}")
print(f"Total images: {len(merged['images'])}")
print(f"Total annotations: {len(merged['annotations'])}")
