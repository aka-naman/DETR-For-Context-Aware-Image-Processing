# Weapon DETR — README

## Project overview
This repository implements a DETR-based object detection pipeline focused on a weapon detection dataset. It contains data preparation, training, evaluation/inspection, and inference utilities. Core model code and training/inference entrypoints live at the repository root.

Key entry files:
- Model & architecture: DETR.py  
- Training entrypoints: train_detr_final_win.py (Windows-oriented training script). There is also a notebook checkpoint: .ipynb_checkpoints/train_detr_win-checkpoint.ipynb  
- Inference & tests: infer_realtime.py, test_inference_image.py, quick_test_inference.py  
- Utilities / inspection: check_weights.py, inspect_detr_outputs.py

Data and artifacts:
- Dataset sources and helpers: weapon_detr/data/  
- Merged dataset files: weapon_detr/data/merged_dataset/annotations.json, train.json, val.json  
- Config: weapon_detr/phase1.yaml  
- Saved models / outputs: outputs/ (e.g. outputs/detr_finetuned_final.pt)

## Typical workflow (step-by-step)
1. Prepare / verify dataset
   - Run merging / fix scripts in weapon_detr/data/ (fix_all_splits.py, fix_category_ids.py, etc.).
   - Confirm merged COCO-like files under weapon_detr/data/merged_dataset/.
2. Configure
   - Edit or verify settings in weapon_detr/phase1.yaml or pass arguments to the training script.
3. Train
   - Start training with the training script, e.g.:
     - train_detr_final_win.py — main training CLI for Windows
     - Notebook (checkpoint): .ipynb_checkpoints/train_detr_win-checkpoint.ipynb
   - Checkpoints and model files are written to outputs/.
4. Validate / Inspect
   - Use inspect_detr_outputs.py to visualize / evaluate saved predictions and checkpoints.
   - Use check_weights.py to verify loaded weights.
5. Inference / Deployment
   - Quick local tests: quick_test_inference.py, test_inference_image.py
   - Real-time demo / webcam: infer_realtime.py
   - Load final model from outputs/detr_finetuned_final.pt

## Workflow diagram (high level)

```text
weapon_detr/data/
       |
       v
[merge / fix scripts]
       |
       v
Merged annotations -> train_detr_final_win.py (training)
       |
       v
   Checkpoints -> outputs/
       |
       +--> inspect_detr_outputs.py (Inspect/Eval)
       |
       +--> infer_realtime.py / test_inference_image.py (Inference)
```

## Quick run examples
- Train (example):
```powershell
python train_detr_final_win.py --config weapon_detr/phase1.yaml
```
- Run a quick image test:
```powershell
python test_inference_image.py --weights outputs/detr_finetuned_final.pt --image path\to\img.jpg
```
- Run realtime demo:
```powershell
python infer_realtime.py --weights outputs/detr_finetuned_final.pt
```

## Dependencies
Create a requirements.txt and install before running:
```powershell
python -m pip install -r requirements.txt
```
Recommended packages (example):
- torch, torchvision
- numpy
- opencv-python
- pycocotools
- Pillow
- matplotlib
- pyyaml
- tqdm

## Notes & troubleshooting
- Confirm which training script to run (train_detr_final_win.py vs any other training entrypoint). Update README if you add/remove scripts.
- Ensure weapon_detr/phase1.yaml paths point to the merged dataset files and outputs folder.
- For GPU training, verify CUDA and matching torch build.
- If inference fails due to model mismatch, run check_weights.py to inspect checkpoint contents.

If you want, I can:
- create requirements.txt in the repo,
- update README in-place,

- or generate a simple troubleshooting checklist specific to errors you see.
