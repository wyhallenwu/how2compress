from ultralytics import YOLO
import torch
import time

inferencer = YOLO("yolov8s.pt", verbose=True).to("cuda")
inferencer.model.eval()
x = torch.randn(1, 3, 96, 480).to("cuda")
start_time = time.time()
output = inferencer.model(x)
end_time = time.time()
print(f"Time taken: {end_time - start_time}")
