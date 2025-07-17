import torch
import cv2
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model.hgmodel import MobileVitV2
from src.model.am_model import AccMpeg
from src.model.utils import ems2selections
from src.utils import image_ops, load, cals, metrics
from src.dataset.dataloader import VisDroneDataset, collate_fn
import os
from ultralytics import YOLO
from tqdm import tqdm
import supervision as sv
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model path")
args = parser.parse_args()
# pretrained/train/exp-accmpeg-1704-0.25q/1-0.0002727616289537327.pth

DEVICE = "cuda:1"
RESIZE_FACTOR = 4
# Hardcode all VisDrone validation sequences
DATASETS = [
    "uav0000086_00000_v",
    "uav0000117_02622_v",
    "uav0000137_00458_v",
    "uav0000182_00000_v",
    "uav0000268_05773_v",
    "uav0000305_00000_v",
    "uav0000339_00001_v"
]
BATCH_SIZE = 1

# Load AM model
model = AccMpeg(1088 // 16, 1920 // 16)
model.load_state_dict(
    torch.load(args.model)
)
model.to(DEVICE)

# Load YOLO model
inferencer = YOLO("yolov8m.pt", verbose=True).to(DEVICE)

# Setup paths
root = "data/visdrone/VisDrone2019-VID-val"
results_root = "/how2compress/results/visdrone"
if not os.path.exists(results_root):
    os.makedirs(results_root)

# Initialize dataset
dataset = VisDroneDataset(
    dataset_dir=root,
    reference_dir="/how2compress/data/VisDroneH264",
    yuv_dir="/how2compress/data/VisDroneYUV",
    resize_factor=RESIZE_FACTOR,
)

model.eval()

# Process each sequence
for seq in DATASETS:
    print(f"\nProcessing sequence: {seq}")
    
    # Setup sequence-specific paths
    r = os.path.join(results_root, f"eval30-45-accmpeg-{seq}-h264.txt")
    rd = os.path.join(results_root, f"decisions30-45-accmpeg-{seq}-h264.txt")
    enc_frames_dir = os.path.join(results_root, seq)
    if not os.path.exists(enc_frames_dir):
        os.makedirs(enc_frames_dir)

    mAPs = []
    mAPs_gt = []
    frames_size = []
    times = []
    ret_frames = []
    decisions = []

    dataset.load_sequence(seq)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False
    )
    height, width = (
        dataset.curr_seq_property["height"],
        dataset.curr_seq_property["width"],
    )
    mb_w, mb_h = cals.macroblocks_wh(width, height)
    resizer = image_ops.resize_img_tensor((mb_h * 4, mb_w * 4))
    count = 0
    
    for images, labels, indices in tqdm(dataloader, desc=f"{seq} val"):
        count += 1
        images = images.to(DEVICE)
        resize_images = resizer(images)
        
        # Model inference
        start_time = time.time()
        ems_map_indices, ems_map_v, selections = model(resize_images)
        end_time = time.time()
        times.append(end_time - start_time)
        
        # Get ground truth detections - using class 0 for person in VisDrone
        targets = inferencer.predict(images, classes=[0,2], verbose=False, save=False)
        targets = metrics.yolo2sv(targets)
        targets = [
            metrics.normalize_detections(
                det,
                (1920, 1080),  # Fixed resolution for resized frames
            )
            for det in targets
        ]
        
        # Process selections
        ret_selections = [[level for _, level in selection] for selection in selections]
        decisions.extend(ret_selections)
        
        # Encode and get compressed frames
        compressed_images, sizes, enc_frames = dataset.enc_and_ret_val(
            indices, selections, DEVICE, qmin=31, qmax=40  # Using AM's QP range
        )
        
        # Get predictions on compressed frames - using class 0 for person in VisDrone
        compressed_images = compressed_images.to(DEVICE)
        preds = inferencer.predict(compressed_images, classes=[0,2], verbose=False, save=False)
        preds = metrics.yolo2sv(preds)
        preds = [
            metrics.normalize_detections(
                det,
                (1920, 1080),  # Fixed resolution for resized frames
            )
            for det in preds
        ]
        
        assert len(preds) == len(
            targets
        ), f"preds size {len(preds)} != targets size {len(targets)}"

        frames_size.extend(sizes)
        mAP_t = sv.MeanAveragePrecision.from_detections(preds, targets)
        mAP = sv.MeanAveragePrecision.from_detections(preds, labels)
        print(f"mAP: {mAP.map50_95}, mAP_t: {mAP_t.map50_95}")
        mAPs.append(mAP)
        mAPs_gt.append(mAP_t)

    # Save results for this sequence
    with open(r, "w") as f:
        for mAP, mAP_gt, frame_size, t in zip(mAPs, mAPs_gt, frames_size, times):
            f.write(
                f"{mAP.map50_95},{mAP.map75},{mAP.map50},{mAP_gt.map50_95},{mAP_gt.map75},{mAP_gt.map50},{frame_size},{t}\n"
            )

    # Save decisions for this sequence
    with open(rd, "w") as f:
        for decision in decisions:
            f.write(",".join(map(str, decision)) + "\n")

    # Save compressed frames for this sequence
    for i, frame in enumerate(ret_frames):
        cv2.imwrite(os.path.join(enc_frames_dir, f"{i+1:06d}.png"), frame) 