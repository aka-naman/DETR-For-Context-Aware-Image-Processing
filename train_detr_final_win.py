"""
train_detr_final_win.py
DETR fine-tuning script with Windows-safe multiprocessing guard.

Requirements:
pip install torch torchvision transformers datasets pycocotools matplotlib tqdm
"""

import os
import json
import math
from pathlib import Path
from collections import defaultdict
import multiprocessing

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------
# CONFIG - EDIT THESE PATHS
# -------------------------
DATA_DIR = r"D:\PROJECT\weapon_detr\data\merged_dataset"
TRAIN_JSON = os.path.join(DATA_DIR, "train_fixed.json")
VAL_JSON = os.path.join(DATA_DIR, "val.json")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

OUTPUT_DIR = r"D:\PROJECT\weapon_detr\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training hyperparams
EPOCHS = 10
BATCH_SIZE = 1                 # reduced from 2 to avoid OOM on RTX 3050
LR = 1e-5
WEIGHT_DECAY = 1e-4
SAVE_EVERY = 1                 # save checkpoint every N epochs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0                # set 0 for Windows if you face issues; increase if desired

# Optional: set to None to infer num classes from JSON
NUM_CLASSES = None

# limit longest edge to reduce memory (tune down if still OOM)
PROCESSOR_SIZE = {"shortest_edge": 600, "longest_edge": 800}


# -------------------------
# HELPER: load categories & infer NUM_CLASSES
# -------------------------
def load_categories_from_json(json_path):
    with open(json_path, "r") as f:
        d = json.load(f)
    cats = d.get("categories", [])
    return cats

# -------------------------
# COCO Dataset class (fixed)
# -------------------------
class CocoDetection(Dataset):
    def __init__(self, img_folder, ann_file, processor, orig_to_label_map):
        with open(ann_file, "r") as f:
            coco = json.load(f)
        self.img_folder = img_folder
        self.processor = processor
        self.images = coco["images"]
        self.annotations = coco.get("annotations", [])
        self.categories = coco.get("categories", [])
        self.orig_to_label_map = orig_to_label_map

        # group annotations by image_id (safely)
        self.image_to_anns = defaultdict(list)
        for ann in self.annotations:
            self.image_to_anns[ann["image_id"]].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        fp = os.path.join(self.img_folder, img_info["file_name"])
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Image not found: {fp}")
        image = Image.open(fp).convert("RGB")

        anns = self.image_to_anns[img_info["id"]]

        ann_list = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            area = ann.get("area", float(w * h))
            iscrowd = ann.get("iscrowd", 0)
            orig_cat = ann["category_id"]
            if orig_cat not in self.orig_to_label_map:
                raise KeyError(f"Found category id {orig_cat} not in mapping.")
            mapped_cat = self.orig_to_label_map[orig_cat]
            ann_list.append({
                "bbox": [x, y, w, h],
                "category_id": mapped_cat,
                "area": area,
                "iscrowd": iscrowd
            })

        target = {
            "image_id": img_info["id"],
            "annotations": ann_list
        }

        # Return the raw PIL image and the target dict.
        # The training loop will call the processor (pad_and_create_pixel_mask / processor(...))
        # so we avoid double-processing and keep tensors creation on the training side.
        return image, target

# -------------------------
# Utilities
# -------------------------
def save_checkpoint(epoch, model, optimizer, out_dir=OUTPUT_DIR):
    path = os.path.join(out_dir, f"detr_epoch{epoch+1}.pt")
    ckpt = {
        "epoch": epoch + 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }
    torch.save(ckpt, path)
    print(f"Saved checkpoint: {path}")

# -------------------------
# MAIN
# -------------------------
def main():
    # load categories and build mapping
    cats = load_categories_from_json(TRAIN_JSON)
    if not cats:
        raise RuntimeError(f"No categories found in {TRAIN_JSON}")
    orig_cat_ids = [c["id"] for c in cats]
    orig_cat_names = {c["id"]: c.get("name", str(c["id"])) for c in cats}
    orig_to_label = {orig_id: idx for idx, orig_id in enumerate(sorted(orig_cat_ids))}
    label_to_name = {v: orig_cat_names[k] for k, v in orig_to_label.items()}

    global NUM_CLASSES
    if NUM_CLASSES is None:
        NUM_CLASSES = len(orig_to_label)
    print(f"Detected categories: {NUM_CLASSES} -> {label_to_name}")

    # load processor/model inside main (so children don't re-exec heavy init)
    print("Loading processor and model...")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=(NUM_CLASSES + 1),   # includes no-object class
        ignore_mismatched_sizes=True
    ).to(DEVICE)
    
        # --- Resume from previous fine-tuned checkpoint if available ---
    resume_path = r"D:\PROJECT\weapon_detr\outputs\detr_finetuned_final.pt"
    if os.path.exists(resume_path):
        print(f"🔁 Resuming training from checkpoint: {resume_path}")
        state_dict = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
    else:
        print("⚠️ No previous checkpoint found, training from scratch.")


    # datasets & loaders
    print("Preparing datasets and dataloaders...")
    train_ds = CocoDetection(IMAGES_DIR, TRAIN_JSON, processor, orig_to_label)
    val_ds = CocoDetection(IMAGES_DIR, VAL_JSON, processor, orig_to_label)

    def collate_fn(batch):
        pixels = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        return pixels, labels

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    print(f"Train size: {len(train_ds)} images | Val size: {len(val_ds)} images")
    print(f"Batch size: {BATCH_SIZE} | Device: {DEVICE} | num_workers: {NUM_WORKERS}")

    # optimizer & amp scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler() if DEVICE=="cuda" else None

    train_losses = []
    val_losses = []

    print("Starting training loop...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        num_batches = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        for pixels, labels in loop:
            # Use the processor new API to encode images AND annotations (limit size to save memory)
            encoded = processor(images=pixels, annotations=labels, size=PROCESSOR_SIZE, return_tensors="pt")
            pixel_values = encoded["pixel_values"].to(DEVICE)
            pixel_mask = encoded.get("pixel_mask")
            if pixel_mask is not None:
                pixel_mask = pixel_mask.to(DEVICE)

            # processor returns encoded["labels"] as a list of dicts with tensors -> move them to device
            encoded_labels = encoded.get("labels")
            if encoded_labels is None:
                raise RuntimeError("Processor did not return 'labels' when encoding annotations.")
            labels_device = []
            for lab in encoded_labels:
                lab2 = {}
                for k, v in lab.items():
                    if isinstance(v, torch.Tensor):
                        lab2[k] = v.to(DEVICE)
                    else:
                        lab2[k] = v
                labels_device.append(lab2)

            optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels_device)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
            else:
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels_device)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

            loss_item = loss.item()
            running_loss += loss_item
            num_batches += 1
            loop.set_postfix(loss=loss_item)

        avg_train_loss = running_loss / max(1, num_batches)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1} TRAIN avg loss: {avg_train_loss:.4f}")

        # validation
        model.eval()
        running_val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            vloop = tqdm(val_loader, desc="  Val", unit="img")
            for v_pixels, v_labels in vloop:
                # use processor to encode validation images + annotations (limit size)
                v_encoded = processor(images=v_pixels, annotations=v_labels, size=PROCESSOR_SIZE, return_tensors="pt")
                v_pixel_values = v_encoded["pixel_values"].to(DEVICE)
                v_pixel_mask = v_encoded.get("pixel_mask")
                if v_pixel_mask is not None:
                    v_pixel_mask = v_pixel_mask.to(DEVICE)

                v_encoded_labels = v_encoded.get("labels")
                if v_encoded_labels is None:
                    raise RuntimeError("Processor did not return 'labels' for validation batch.")
                v_labels_device = []
                for lab in v_encoded_labels:
                    lab2 = {}
                    for k, v in lab.items():
                        if isinstance(v, torch.Tensor):
                            lab2[k] = v.to(DEVICE)
                        else:
                            lab2[k] = v
                    v_labels_device.append(lab2)

                outputs = model(pixel_values=v_pixel_values, pixel_mask=v_pixel_mask, labels=v_labels_device)
                vloss = outputs.loss
                running_val_loss += vloss.item()
                val_batches += 1

        avg_val_loss = running_val_loss / max(1, val_batches)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1} VAL avg loss: {avg_val_loss:.4f}")

        if (epoch + 1) % SAVE_EVERY == 0:
            save_checkpoint(epoch, model, optimizer, out_dir=OUTPUT_DIR)

    # final save and plot
    final_path = os.path.join(OUTPUT_DIR, "detr_finetuned_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to: {final_path}")

    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label="train")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plot_path = os.path.join(OUTPUT_DIR, "loss.png")
    plt.savefig(plot_path)
    print(f"Loss plot saved to: {plot_path}")

    print("Training complete.")

# -------------------------
# Entrypoint for Windows
# -------------------------
if __name__ == "__main__":
    # required for frozen executables; safe to call on Windows
    multiprocessing.freeze_support()
    # Optional: explicitly set spawn start method (Windows default is spawn)
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # start method already set
        pass
    main()
