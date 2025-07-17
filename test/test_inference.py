from ultralytics import YOLO
from src.dataset.mot_utils import get_MOT_GT
from src.dataset.panda_utils import get_panda_GT
from src.utils.load import load_h264_training
from tqdm import tqdm
import os
import supervision as sv
from src.utils import metrics
from src.utils import image_ops
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seq", type=str, help="sequence name")
args = parser.parse_args()
# evaluation
# MOT17-04: Tensor input
# 30: mAP50_95: 0.606149001554652, mAP75: 0.7093165379267867, mAP50: 0.7741118599874512
# 34: mAP50_95: 0.5889753821189325, mAP75: 0.6905192397652884, mAP50: 0.7563671020884795
# 37: mAP50_95: 0.5660094185995332, mAP75: 0.6611814044772828, mAP50: 0.7326189097704151
# 41: mAP50_95: 0.5176869835559059, mAP75: 0.6027643119414224, mAP50: 0.6774497667990049

yolo = YOLO("/how2compress/pretrained/best_MOT_1920.pt")
gt = get_MOT_GT("/how2compress/data/MOT17Det/train", [1])
root = "/how2compress/data/detections"
seq = args.seq
# yolo = YOLO("/how2compress/pretrained/best-panda-2560.pt")
# gt = get_panda_GT("/how2compress/data/pandas/unzipped/train_annos")
# root = "/how2compress/data/pandasH264"
# seq = "01_University_Canteen"
qp = 37
last = 400
DEVICE = "cuda:1"

path = os.path.join(root, seq, str(qp))
frames = sorted(os.listdir(path))
transform = image_ops.vit_transform_fn()
detections = []
for frame in tqdm(frames):
    frame_path = os.path.join(path, frame)
    img = load_h264_training(frame_path)
    img = image_ops.wrap_img(img)
    img = transform(img)
    img = img.unsqueeze(0).to(DEVICE)
    results = yolo.predict(img, classes=[1], device=DEVICE, verbose=False)
    results = metrics.yolo2sv(results)
    results = [metrics.normalize_detections(result, (1920, 1080)) for result in results]

    detections.extend(results)

targets = gt[seq]
assert len(detections) == len(targets), f"{len(detections)} != {len(targets)}"
mAP = sv.MeanAveragePrecision.from_detections(detections, targets)
print(f"qp {qp}, mAP50_95: {mAP.map50_95}, mAP75: {mAP.map75}, mAP50: {mAP.map50}")
