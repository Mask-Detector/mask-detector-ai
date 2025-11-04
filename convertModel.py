import sys
import torch

# tambahkan path ke YOLOv5 lokal
sys.path.insert(0, './yolov5')

from models.experimental import attempt_load

weights = "yolov5/best.pt"

print("Loading model dengan YOLOv5...")
model = attempt_load(weights, device="cpu")  # langsung pakai device

# Simpan ulang model jadi clean checkpoint
torch.save({"model": model}, "models/best_converted.pt")
print("✅ Model berhasil dikonversi ke models/best_converted.pt")
