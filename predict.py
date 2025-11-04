import requests
import cv2
import numpy as np
import torch

# alamat ESP32
ESP32_IP = "http://192.168.0.102"

# load model YOLOv5
print("Memuat model YOLOv5...")
model = torch.hub.load("ultralytics/yolov5", "custom", path="models/mask_yolov5.pt")

while True:
    try:
        # ambil snapshot dari ESP32
        resp = requests.get(f"{ESP32_IP}/capture", timeout=10)
        img_arr = np.frombuffer(resp.content, np.uint8)
        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        if frame is None:
            print("⚠️ Gagal decode gambar dari ESP32")
            continue

        # prediksi YOLO
        results = model(frame)

        # ambil label hasil deteksi
        labels = results.pandas().xyxy[0]["name"].tolist()
        print("Hasil deteksi:", labels)

        # render hasil deteksi
        img_rendered = np.squeeze(results.render())

        # tampilkan di window
        cv2.imshow("Deteksi Masker", img_rendered)

    except Exception as e:
        print("❌ Error:", e)

    # tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
