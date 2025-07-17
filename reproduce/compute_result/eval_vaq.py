import torch
import cv2
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model.hgmodel import MobileVitV2
from src.model.utils import ems2selections
from src.utils import image_ops, load, cals, metrics
from src.dataset.dataloader import MOTDataset, collate_fn
from src.dataset.mot_utils import get_MOT_GT
import os
from ultralytics import YOLO
from tqdm import tqdm
import supervision as sv
import time
import numpy as np

import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--dataset", type=str, help="dataset name")

# args = parser.parse_args()
DEVICE = "cuda:1"
DATASET = ["MOT17-02", "MOT17-04", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]
# DATASET = ["MOT17-04"]
BATCH_SIZE = 1


gt = get_MOT_GT("/how2compress/data/MOT17Det/train", [1])

transform = image_ops.vit_transform_fn()
# root = "/how2compress/data/MOT17H264VAQ"
root = "/how2compress/data/MOT17H264VAQ0-2"
results_root = "/how2compress/results"
r = os.path.join(results_root + "eval-MOT17-VAQ0-2.txt")


inferencer = YOLO("/how2compress/pretrained/best_MOT_1920.pt", verbose=False)

for dataset in DATASET:
    mAPs = []
    mAPs_gt = []
    frames_size = []
    times = []
    ret_frames = []
    decisions = []
    path = os.path.join(root, dataset)
    frames = sorted(os.listdir(path))
    frames = [os.path.join(path, frame) for frame in frames]
    gtdataset = gt[dataset]
    for i, frame in tqdm(enumerate(frames)):
        img = load.load_h264_training(frame)
        img = image_ops.wrap_img(img)
        img = transform(img)
        img = img.unsqueeze(0).to(DEVICE)
        results = inferencer.predict(img, classes=[1], device=DEVICE, verbose=False)
        results = metrics.yolo2sv(results)
        results = [
            metrics.normalize_detections(result, (1920, 1080)) for result in results
        ]

        label = gtdataset[i]
        filesize = os.path.getsize(frame)
        mAP = sv.MeanAveragePrecision.from_detections(results, [label])
        # print(mAP.map50_95, filesize / 1024)
        mAPs.append(mAP)
        frames_size.append(filesize / 1024)
        # print(f"frame: {frame}, filesize: {filesize/1024}")

    mean_map50_95 = np.mean([mAP.map50_95 for mAP in mAPs])
    mean_map75 = np.mean([mAP.map75 for mAP in mAPs])
    mean_map50 = np.mean([mAP.map50 for mAP in mAPs])
    sig_map50_95 = np.std([mAP.map50_95 for mAP in mAPs])
    sig_map75 = np.std([mAP.map75 for mAP in mAPs])
    sig_map50 = np.std([mAP.map50 for mAP in mAPs])
    mean_frames_size = np.mean(frames_size)
    print(
        f"{dataset}, mAP50_95: {mean_map50_95}, {sig_map50_95}, mAP75: {mean_map75}, {sig_map75}, mAP50: {mean_map50}, {sig_map50}, mean_frames_size: {mean_frames_size}"
    )
    with open(r, "a") as f:
        f.write(
            f"{dataset}, mAP50_95: {mean_map50_95}, {sig_map50_95}, mAP75: {mean_map75}, {sig_map75}, mAP50: {mean_map50}, {sig_map50}, mean_frames_size: {mean_frames_size}\n"
        )
