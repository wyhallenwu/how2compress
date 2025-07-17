import torch
import cv2
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model.hgmodel import MobileVitV2
from src.model.utils import ems2selections
from src.model.am_model import AccMpeg
from src.utils import image_ops, load, cals, metrics
from src.dataset.dataloader import MOTDataset, collate_fn
import os
from ultralytics import YOLO
from tqdm import tqdm
import supervision as sv
import time
import argparse
import subprocess
import numpy as np
import math
import tempfile
import shutil

mapping = {0: 45, 1: 41, 2: 37, 3: 34, 4: 30}

num = 25
aq_mode = 1

DATASET = ["MOT17-02", "MOT17-04", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]
# uni qp
for dataset in DATASET:
    dest = f"/how2compress/data/UNI30CHUNK/{dataset}"
    root = f"/how2compress/data/MOT17Det/train/{dataset}/img1"
    files = os.listdir(root)
    num_frames = len(files)
    if not os.path.exists(dest):
        os.makedirs(dest)

    num_chunk = math.ceil(num_frames / num)

    for i in tqdm(range(num_chunk)):
        start_idx = i * num + 1
        with tempfile.TemporaryDirectory() as tmpdirname:
            for j in range(num):
                idx = start_idx + j
                if idx > num_frames:
                    break
                shutil.copyfile(
                    os.path.join(root, f"{idx:06d}.jpg"),
                    os.path.join(tmpdirname, f"{j:06d}.jpg"),
                )
            subprocess.run(
                [
                    "/myh264/bin/ffmpeg",
                    "-y",
                    "-i",
                    f"{tmpdirname}/%06d.jpg",
                    "-start_number",
                    str(0),
                    "-vframes",
                    str(num),
                    "-framerate",
                    str(num),
                    "-qp",
                    "30",
                    "-pix_fmt",
                    "yuv420p",
                    os.path.join(dest, f"uni-{(i+1):02d}.mp4"),
                ]
            )

            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    os.path.join(dest, f"uni-{(i+1):02d}.mp4"),
                    "-c:v",
                    "libx264",
                    "-qp",
                    "30",
                    "-x264-params",
                    f"aq-mode={aq_mode}",
                    os.path.join(dest, f"aq-{aq_mode}-{(i+1):02d}.mp4"),
                ]
            )
