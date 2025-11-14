# inspect_detr_outputs.py
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image
import os

MODEL_PATH = r"D:\PROJECT\weapon_detr\outputs\detr_finetuned_final.pt"
IMAGE_PATH = r"D:\PROJECT\weapon_detr\data\merged_dataset\images\gun.jpg"  # change to your test image
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model + processor...")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=3,             # (your classes + 1 background); adjust if you changed NUM_CLASSES
    ignore_mismatched_sizes=True
)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state, strict=False)
model.to(DEVICE)
model.eval()

image = Image.open(IMAGE_PATH).convert("RGB")
inputs = processor(images=image, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    outputs = model(**inputs)

# logits shape and sample
logits = outputs.logits  # (batch, queries, num_labels)
boxes = outputs.pred_boxes  # (batch, queries, 4) in cx,cy,w,h normalized (0..1)

print("logits.shape:", logits.shape)
print("boxes.shape:", boxes.shape)

probs = logits.softmax(-1)  # softmax over classes
top_probs, top_labels = probs.max(-1)  # (batch, queries)
top_probs = top_probs[0].cpu()
top_labels = top_labels[0].cpu()
pred_boxes = boxes[0].cpu()

no_object_idx = logits.shape[-1] - 1
print(f"Assuming 'no-object' is index {no_object_idx} (last index).")

nonbg_count = int((top_labels != no_object_idx).sum().item())
print(f"Top-label non-background count: {nonbg_count} / {top_labels.shape[0]}\n")

# print first 50 queries for inspection
for i in range(min(50, top_labels.shape[0])):
    print(f"Q{i:03d}  label={int(top_labels[i].item())}  prob={top_probs[i].item():.4f}  box(norm)={pred_boxes[i].tolist()}")

# Try HF post_process with a very low threshold so we can see everything it returns
from torch import tensor
target_sizes = torch.tensor([image.size[::-1]]).to(DEVICE)
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.01)[0]
print("\nPost-processed (threshold=0.01) results count:", len(results["scores"]))
for i, (s,l,b) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
    print(f"-> {i}: score={s.item():.4f} label={int(l.item())} box={ [round(x,2) for x in b.tolist()] }")
