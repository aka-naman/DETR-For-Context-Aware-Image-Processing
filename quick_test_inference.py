# quick_test_inference.py
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image, ImageDraw, ImageFont
import os

MODEL_PATH = r"D:\PROJECT\weapon_detr\outputs\detr_finetuned_final.pt"
IMAGE_PATH = r"D:\PROJECT\weapon_detr\data\merged_dataset\images\gun.jpg"  # change to an image that clearly shows a knife or gun
SAVE_PATH = r"D:\PROJECT\weapon_detr\outputs\prediction_debug_final.jpg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# label mapping: 0 -> knife, 1 -> gun (matches your train_fixed.json)
ID2LABEL = {0: "knife", 1: "gun"}
CONF_THRESHOLD = 0.25  # debug low threshold

print("Loading model + processor...")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=len(ID2LABEL) + 1,
    ignore_mismatched_sizes=True
)
state = torch.load(MODEL_PATH, map_location=DEVICE)
# your final file is a raw state_dict -> load directly
model.load_state_dict(state, strict=False)
model.to(DEVICE)
model.eval()

image = Image.open(IMAGE_PATH).convert("RGB")
inputs = processor(images=image, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    outputs = model(**inputs)

# print confidence info
max_conf = outputs.logits.softmax(-1).max(-1).values.max()
print(f"Max confidence across all predictions: {max_conf.item():.4f}")

# post-process
target_sizes = torch.tensor([image.size[::-1]]).to(DEVICE)
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
print(f"Raw predictions: {len(results['scores'])}")

# print first 20 predictions (score,label,box)
for i, (s, l, b) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
    if i >= 20: break
    print(f"[{i}] score={s.item():.4f} label={l.item()} box={[round(x,2) for x in b.tolist()]}")

# draw all above threshold
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()
found = False
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    if score < CONF_THRESHOLD:
        continue
    found = True
    label_id = int(label.item())
    # HF DETR leaves labels as-is (we used 0..C-1 during training). Use ID directly.
    label_name = ID2LABEL.get(label_id, f"id:{label_id}")
    x1, y1, x2, y2 = [int(round(x)) for x in box.tolist()]
    draw.rectangle([(x1,y1),(x2,y2)], outline="red", width=3)
    draw.text((x1, max(0, y1-12)), f"{label_name} {score:.2f}", fill="red", font=font)

image.save(SAVE_PATH)
print(f"Saved annotated image to: {SAVE_PATH}")
print("Found detections above threshold." if found else "No detections above threshold; try lowering CONF_THRESHOLD or using a clearer test image.")
