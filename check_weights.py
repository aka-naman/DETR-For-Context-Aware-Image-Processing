import torch

path = r"D:\PROJECT\weapon_detr\outputs\detr_finetuned_final.pt"
ckpt = torch.load(path, map_location="cpu")

print("✅ Keys in checkpoint:", len(ckpt))
if isinstance(ckpt, dict) and "model_state" in ckpt:
    print("⚙️  This is a full checkpoint (with optimizer). Use ckpt['model_state'] for loading.")
else:
    print("🧩 This is a raw model.state_dict() — load it directly.")
