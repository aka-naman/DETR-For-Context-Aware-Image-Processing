import os
import json
import shutil
from sklearn.model_selection import train_test_split

# ==============================
# CONFIG
# ==============================

# Paths (edit if needed)
merged_json = r"D:\PROJECT\weapon_detr\data\merged_annotations.json"
dataset_a_images = r"D:\PROJECT\weapon_detr\data\dataset_A\train"
dataset_b_images = r"D:\PROJECT\weapon_detr\data\dataset_B\train"

output_dir = r"D:\PROJECT\weapon_detr\data\merged_dataset"

# Train/val split ratio
val_ratio = 0.2

# ==============================
# CREATE OUTPUT FOLDERS
# ==============================
images_dir = os.path.join(output_dir, "images")
os.makedirs(images_dir, exist_ok=True)

# ==============================
# 1️⃣ COPY IMAGES FROM BOTH DATASETS
# ==============================
print("📁 Copying images from dataset_A and dataset_B...")

def copy_images(src_dir, dst_dir):
    count = 0
    for fname in os.listdir(src_dir):
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        if os.path.isfile(src_path):
            shutil.copy(src_path, dst_path)
            count += 1
    return count

count_a = copy_images(dataset_a_images, images_dir)
count_b = copy_images(dataset_b_images, images_dir)
print(f"✅ Copied {count_a} images from dataset_A and {count_b} from dataset_B")

# ==============================
# 2️⃣ LOAD MERGED ANNOTATIONS
# ==============================
print("📄 Loading merged_annotations.json...")
with open(merged_json, "r") as f:
    merged = json.load(f)

images = merged["images"]
annotations = merged["annotations"]
categories = merged["categories"]

# ==============================
# 3️⃣ SPLIT INTO TRAIN / 
# ==============================
print("✂️ Splitting dataset into train/val...")
train_imgs, val_imgs = train_test_split(images, test_size=val_ratio, random_state=42)

train_ids = {img["id"] for img in train_imgs}
val_ids = {img["id"] for img in val_imgs}

def filter_anns(img_ids):
    return [ann for ann in annotations if ann["image_id"] in img_ids]

train_anns = filter_anns(train_ids)
val_anns = filter_anns(val_ids)

train_json = {
    "images": train_imgs,
    "annotations": train_anns,
    "categories": categories,
    "info": {"description": "train split"}
}

val_json = {
    "images": val_imgs,
    "annotations": val_anns,
    "categories": categories,
    "info": {"description": "validation split"}
}

# ==============================
# 4️⃣ SAVE JSON FILES
# ==============================
train_path = os.path.join(output_dir, "train.json")
val_path = os.path.join(output_dir, "val.json")
full_path = os.path.join(output_dir, "annotations.json")

os.makedirs(output_dir, exist_ok=True)
with open(train_path, "w") as f: json.dump(train_json, f, indent=2)
with open(val_path, "w") as f: json.dump(val_json, f, indent=2)
shutil.copy(merged_json, full_path)

print(f"\n✅ JSONs saved successfully:")
print(f"  → {train_path}")
print(f"  → {val_path}")

# ==============================
# 5️⃣ VERIFY COCO INTEGRITY
# ==============================
print("\n🔍 Verifying dataset integrity...")

missing_images = []
for img in images:
    img_path = os.path.join(images_dir, img["file_name"])
    if not os.path.exists(img_path):
        missing_images.append(img["file_name"])

if missing_images:
    print(f"⚠️ Found {len(missing_images)} missing image(s):")
    for m in missing_images[:10]:
        print(f"   - {m}")
    print("⚠️ Please ensure these files exist in your merged 'images/' folder.")
else:
    print("✅ All image files are present and correctly referenced in annotations.")

# ==============================
# 6️⃣ SUMMARY
# ==============================
print("\n📊 Summary:")
print(f"  Total images: {len(images)}")
print(f"  Train split: {len(train_imgs)} images, {len(train_anns)} annotations")
print(f"  Val split:   {len(val_imgs)} images, {len(val_anns)} annotations")
print(f"\n🗂️ Final structure:")
print(f"  {output_dir}\\")
print("    ├── images\\")
print("    ├── annotations.json")
print("    ├── train.json")
print("    └── val.json")
print("\n🚀 Dataset is ready for DETR fine-tuning!")
