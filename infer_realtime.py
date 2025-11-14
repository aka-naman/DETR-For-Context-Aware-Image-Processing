"""
infer_realtime.py
Real-time inference for fine-tuned DETR (knife/gun detection)
--------------------------------------------------------------
Requires:
    pip install torch torchvision transformers opencv-python numpy
Optional:
    pip install playsound  (for Phase 4 alert)
"""

import cv2
import torch
import numpy as np
from transformers import DetrForObjectDetection, DetrImageProcessor
from pathlib import Path

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = r"D:\PROJECT\weapon_detr\outputs\detr_finetuned_final.pt"
LABELS = {0: "knife", 1: "gun"}       # your trained classes
THRESHOLD = 0.7                       # confidence threshold
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_SOURCE = 0                      # 0 = webcam; or "video.mp4"

# -------------------------
# LOAD MODEL + PROCESSOR
# -------------------------
print("Loading model and processor...")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=len(LABELS) + 1,
    ignore_mismatched_sizes=True
)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)
model.to(DEVICE)
model.eval()
print("✅ Model loaded successfully.\n")

# -------------------------
# VIDEO LOOP
# -------------------------
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise RuntimeError("❌ Could not open video source.")

print("🎥 Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR → RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess for DETR
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Postprocess to get boxes, scores, labels
    target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    # for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    #     if score < THRESHOLD:
    #         continue
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        print(f"→ Detected label {label.item()} | score={score:.3f}")
        if score < THRESHOLD:
            continue

        label_id = label.item()
        label_name = LABELS.get(label_id, f"id_{label_id}")
        box = [round(i, 1) for i in box.tolist()]
        x1, y1, x2, y2 = map(int, box)

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label_name}: {score:.2f}"
        cv2.putText(frame, text, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Weapon Detection - DETR", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Exited gracefully.")
