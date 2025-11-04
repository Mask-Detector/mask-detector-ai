import torch
import cv2
import sys
import os

# tambahkan path YOLOv5 lokal (pastikan folder ./yolov5 ada di proyek)
sys.path.append(os.path.join(os.getcwd(), "yolov5"))

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

# path model hasil training
weights = "yolov5/best.pt"
device = select_device('0')  # pakai 'cpu' kalau tanpa GPU
imgsz = 640

# load model
model = attempt_load(weights, device=device)
stride = int(model.stride.max())
imgsz = check_img_size(imgsz, s=stride)

# buka webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # preprocessing frame -> letterbox resize
    img = letterbox(frame, imgsz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR ke RGB, to 3xHxW
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # inference
    pred = model(img)[0]
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    # render deteksi ke frame asli
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                label = f"{int(cls)} {conf:.2f}"
                # kotak deteksi
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])),
                                     (int(xyxy[2]), int(xyxy[3])),
                                     (0, 255, 0), 2)
                cv2.putText(frame, label,
                            (int(xyxy[0]), int(xyxy[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

    # tampilkan hasil
    cv2.imshow("Webcam Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()