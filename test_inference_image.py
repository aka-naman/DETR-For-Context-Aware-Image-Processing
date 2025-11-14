import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image, ImageDraw, ImageFont
import os

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = r"D:\PROJECT\weapon_detr\outputs\detr_finetuned_final.pt"
IMAGE_PATH = r"D:\PROJECT\weapon_detr\data\merged_dataset\images\gun.jpg"  # <-- change this
SAVE_PATH = r"D:\PROJECT\weapon_detr\outputs\prediction_debug.jpg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABELS = {0: "knife", 1: "gun"}
CONF_THRESHOLD = 0.3  # lower to visualize everything

# -------------------------
# LOAD MODEL & PROCESSOR
# -------------------------
print("Loading fine-tuned DETR model...")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=len(LABELS) + 1,
    ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
model.to(DEVICE)
model.eval()

# -------------------------
# INFERENCE
# -------------------------
image = Image.open(IMAGE_PATH).convert("RGB")
inputs = processor(images=image, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    outputs = model(**inputs)

# Convert outputs to COCO format
target_sizes = torch.tensor([image.size[::-1]]).to(DEVICE)  # (H, W)
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

print(f"🔍 Found {len(results['scores'])} raw predictions")

# Debug print all predictions
for i, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
    print(f"[{i}] {LABELS.get(label.item(), 'unknown')} - conf: {score.item():.4f} - box: {box.tolist()}")

# -------------------------
# DRAW BOXES
# -------------------------
draw = ImageDraw.Draw(image)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    if score < CONF_THRESHOLD:
        continue
    label_name = LABELS.get(label.item(), "unknown")
    box = [round(i, 2) for i in box.tolist()]
    x_min, y_min, x_max, y_max = box
    draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=3)
    text = f"{label_name} {score:.2f}"
    draw.text((x_min, y_min - 10), text, fill="red")

image.save(SAVE_PATH)
print(f"✅ Saved result with bounding boxes to: {SAVE_PATH}")
