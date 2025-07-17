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
import tempfile
import shutil

# mapping = {0: 45, 1: 43, 2: 37, 3: 34, 4: 30}
mapping = {0: 45, 1: 43, 2: 37, 3: 34, 4: 30}
dataset = "MOT17-13"
src_root = f"/code/how2compress/data/MOT17YUVCHUNK25/{dataset}/"
videos = sorted(os.listdir(src_root))
num_videos = len(videos)
accmpeg_model = (
    # "/how2compress/pretrained/train/exp-accmpeg-1702-0.35q/1-0.0002956777562219681.pth"
    # "/how2compress/pretrained/train/exp-accmpeg-1704-0.35q/1--0.007357695757508331.pth"
    # "/how2compress/pretrained/train/exp-accmpeg-1709-0.45q/0--0.0059172190316897355.pth"
    # "/how2compress/pretrained/train/exp-accmpeg-1710-0.45q/0--0.009722388405215221.pth"
    # "/how2compress/pretrained/train/exp-accmpeg-1711-0.45q/0--0.011230770333203854.pth"
    "/code/how2compress/pretrained/train/exp-accmpeg-1713-0.45q/0--0.03347167812188889.pth"
)

# ours_model = "/how2compress/pretrained/train/exp1702-1-1/1-0.4764635543758309+-0.001443803332996929-0.988-0.921.pth"
# ours_model = "/how2compress/pretrained/train/exp1704-1/1-0.5303217708720351+-0.0042760108615707-0.993-0.978.pth"
# ours_model = "/how2compress/pretrained/train/exp1709-1/1-0.5740472400763013+-0.007589454466788492-0.982-0.931.pth"
# ours_model = "/how2compress/pretrained/train/exp1710-1/1-0.4620776571761211+-0.0023735605196651965-0.993-0.971.pth"
# ours_model = "/how2compress/pretrained/train/exp1711-1-1/1-0.5398975733473949+0.0023943903714293002-0.991-0.947.pth"
ours_model = "/code/how2compress/pretrained/train/exp1713-1-1/1-0.3470107100380082+-0.01633170276732193-0.991-0.952.pth"

root_dir = f"video-result/{dataset}"
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

raw = f"/code/how2compress/data/MOT17Det/train/{dataset}/img1"

num = 25
bf = 3
num_frames = len(os.listdir(raw))
# start_idx = 1

DEVICE = "cuda"
RESIZE_FACTOR = 4
BATCH_SIZE = 1

mb_w, mb_h = cals.macroblocks_wh(1920, 1080)
transform = image_ops.vit_transform_fn()
resizer = image_ops.resize_img_tensor((mb_h * 4, mb_w * 4))

# accmpeg
model1 = AccMpeg(mb_h, mb_w)
model1.load_state_dict(torch.load(accmpeg_model))
model1.to(DEVICE)

model2 = MobileVitV2()
model2.load_state_dict(torch.load(ours_model))
model2.to(DEVICE)
model2.set_output_size((mb_h, mb_w))


for idx, video in tqdm(enumerate(videos)):
    start_idx = 1 + idx * num
    video = os.path.join(src_root, video)

    # video = f"/how2compress/data/MOT17DetYUV/{dataset}/000001.yuv"

    accmpeg_dest_path = f"nv-accmpeg-{(idx+1):02d}.h264"

    ours_dest_path = f"nv-ours-{(idx+1):02d}.h264"

    uniqp_dest_path = f"nv-uniqp-{(idx+1):02d}.h264"
    spatial_aq_dest_path = f"nv-aq-{(idx+1):02d}.h264"

    with open("/myh264/qp_matrix_file", "w") as f:
        for i in range(num):
            image = cv2.imread(os.path.join(raw, f"{(i+start_idx):06d}.jpg"))
            image = image_ops.wrap_img(image)
            image = transform(image).unsqueeze(0).to(DEVICE)
            # print(image.shape)
            resize_image = resizer(image)
            ems_map_indices, ems_map_v, selections = model1(resize_image)
            selections = [mapping[i] for _, i in selections[0]]

            values, counts = np.unique(selections, return_counts=True)
            print(f"h264 - values: {values}, counts: {counts}")

            matrix = np.reshape(selections, (mb_h, mb_w))
            for row in matrix:
                f.write(" ".join(map(str, row)) + "\n")

    with tempfile.TemporaryDirectory() as tmpdirname:
        for j in range(num):
            idx1 = start_idx + j
            if idx1 > num_frames:
                break
            shutil.copyfile(
                os.path.join(raw, f"{idx1:06d}.jpg"),
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
                "25",
                "-qp",
                "20",
                "-pix_fmt",
                "yuv420p",
                os.path.join(root_dir, f"h264-accmpeg-{(idx+1):02d}.mp4"),
            ]
        )

        # ours
        with open("/myh264/qp_matrix_file", "w") as f:
            for i in range(num):
                # count += 1
                image = cv2.imread(os.path.join(raw, f"{(i+start_idx):06d}.jpg"))
                image = image_ops.wrap_img(image)
                image = transform(image).unsqueeze(0).to(DEVICE)
                # print(image.shape)
                resize_image = resizer(image)
                ems_map_indices, ems_map_v, selections = model2(resize_image)
                # print(selections[0])
                selections = [mapping[i] for _, i in selections[0]]

                values, counts = np.unique(selections, return_counts=True)
                print(f"h264 - values: {values}, counts: {counts}")

                matrix = np.reshape(selections, (mb_h, mb_w))
                for row in matrix:
                    f.write(" ".join(map(str, row)) + "\n")

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
                "25",
                "-qp",
                "10",
                "-pix_fmt",
                "yuv420p",
                os.path.join(root_dir, f"h264-ours-{(idx+1):02d}.mp4"),
            ]
        )

    exe = "/how2compress/src/tools/AppEncCudaEM"
    exe1 = "/how2compress/src/tools/AppEncCudaNoEM"

    # uni qp
    result1 = subprocess.run(
        [
            exe1,
            "-i",
            video,
            "-o",
            os.path.join(root_dir, uniqp_dest_path),
            "-s",
            "1920x1080",
            "-gpu",
            "0",
            "-qmin",
            "30",
            "-gop",
            str(num),
            "-qmax",
            "30",
            "-bf",
            str(bf),
            "-fps",
            str(num),
            "-constqp",
            "30",
            "-initqp",
            "30",
            "-tuninginfo",
            "ultralowlatency",
            "-rc",
            "constqp",
        ]
    )

    # spatial AQ
    subprocess.run(
        [
            exe1,
            "-i",
            video,
            "-o",
            os.path.join(root_dir, spatial_aq_dest_path),
            "-s",
            "1920x1080",
            "-gop",
            str(num),
            "-qmin",
            "30",
            "-qmax",
            "45",
            "-bf",
            str(bf),
            "-fps",
            str(num),
            "-aq",
            "0",
            "-initqp",
            "35",
            "-gpu",
            "0",
            "-tuninginfo",
            "ultralowlatency",
            "-rc",
            "cbr",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    # accmpeg
    with open("test_bitrate_emphasis.txt", "w") as f:
        for i in range(num):
            image = cv2.imread(os.path.join(raw, f"{(i+start_idx):06d}.jpg"))
            image = image_ops.wrap_img(image)
            image = transform(image).unsqueeze(0).to(DEVICE)
            # print(image.shape)
            resize_image = resizer(image)
            ems_map_indices, ems_map_v, selections = model1(resize_image)

            sss = [level for _, level in selections[0]]
            values, counts = np.unique(sss, return_counts=True)
            print(f"values: {values}, counts: {counts}")

            f.write(",".join([str(level) for _, level in selections[0]]) + "\n")

    result = subprocess.run(
        [
            exe,
            "-i",
            video,
            "-o",
            os.path.join(root_dir, accmpeg_dest_path),
            "-s",
            "1920x1080",
            "-gpu",
            "0",
            "-e",
            "test_bitrate_emphasis.txt",
            "-gop",
            str(num),
            "-qmin",
            "30",
            "-qmax",
            "45",
            "-bf",
            str(bf),
            "-fps",
            str(num),
            "-constqp",
            "45",
            "-initqp",
            "45",
            "-tuninginfo",
            "ultralowlatency",
            "-rc",
            "constqp",
        ]
    )

    with open("test_bitrate_emphasis.txt", "w") as f:
        for i in range(num):
            image = cv2.imread(os.path.join(raw, f"{(i+start_idx):06d}.jpg"))
            image = image_ops.wrap_img(image)
            image = transform(image).unsqueeze(0).to(DEVICE)
            # print(image.shape)
            resize_image = resizer(image)
            ems_map_indices, ems_map_v, selections = model2(resize_image)

            sss = [level for _, level in selections[0]]
            values, counts = np.unique(sss, return_counts=True)
            print(f"values: {values}, counts: {counts}")

            f.write(",".join([str(level) for _, level in selections[0]]) + "\n")

    result = subprocess.run(
        [
            exe,
            "-i",
            video,
            "-o",
            os.path.join(root_dir, ours_dest_path),
            "-s",
            "1920x1080",
            "-gpu",
            "0",
            "-e",
            "test_bitrate_emphasis.txt",
            "-gop",
            str(num),
            "-qmin",
            "30",
            "-qmax",
            "45",
            "-bf",
            str(bf),
            "-fps",
            str(num),
            "-constqp",
            "45",
            "-initqp",
            "45",
            "-tuninginfo",
            "ultralowlatency",
            "-rc",
            "constqp",
        ]
    )
