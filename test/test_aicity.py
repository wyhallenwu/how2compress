from ultralytics import YOLO
import os
import cv2
from src.utils import image_ops, metrics, cals


DEVICE = "cuda:0"

inferencer = YOLO("yolov8s.pt", verbose=True).to(DEVICE)
cap = cv2.VideoCapture("/how2compress/data/aicity/train/S01/c001/vdo.avi")
while True:
    ret, frame = cap.read()
    # frame = image_ops.wrap_img(frame)
    # frame = image_ops.vit_transform_fn()(frame).unsqueeze(0).to(DEVICE)
    # print(frame.shape)
    if not ret:
        break
    result = inferencer.predict(
        frame,
        imgsz=(1088, 1920),
        save=True,
        save_txt=True,
        classes=[2, 7],
        device=DEVICE,
    )

# inferencer.predict(
#     "/how2compress/data/aicity/train/S01/c001/vdo.avi",
#     save=True,
#     save_txt=True,
#     imgsz=(1088, 1920),
#     classes=[2, 7],
#     device=DEVICE,
# )
