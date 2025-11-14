from transformers import DetrForObjectDetection, DetrImageProcessor

model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

print("✅ DETR loaded successfully!")
