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

# mapping = {0: 45, 1: 43, 2: 37, 3: 34, 4: 30}
mapping = {0: 45, 1: 43, 2: 37, 3: 34, 4: 30}
dataset = "MOT17-02"

video = f"/how2compress/data/MOT17YUVCHUNK/{dataset}/000001.yuv"
root_dir = "video-result"
# video = f"/how2compress/data/MOT17DetYUV/{dataset}/000001.yuv"
raw = f"/how2compress/data/MOT17Det/train/{dataset}/img1"
accmpeg_model = (
    # "/how2compress/pretrained/train/exp-accmpeg-1702-0.25q/1-0.0031555224912009905.pth"
    # "/how2compress/pretrained/train/exp-accmpeg-1704-0.25q/1-0.0002727616289537327.pth"
    # "/how2compress/pretrained/train/exp-accmpeg-1709-0.25q/1-0.0022972747083872536.pth"
    # "/how2compress/pretrained/train/exp-accmpeg-1710-0.25q/1-2.0321853926863476e-05.pth"
    # "/how2compress/pretrained/train/exp-accmpeg-1711-0.25q/1--0.002179703994586335.pth"
    # "/how2compress/pretrained/train/exp-accmpeg-1713-0.25q/1-0.00327935070394747.pth"
)
accmpeg_dest_path = f"test_bitrate_chunk-accmpeg-{dataset}.h264"

ours_model = "/how2compress/pretrained/train/exp1702-1-1/1-0.4764635543758309+-0.001443803332996929-0.988-0.921.pth"
# ours_model = "/how2compress/pretrained/train/exp1704-n3/5-0.5452541371035616+-0.007466545299626204-0.993-0.965.pth"
# ours_model = "/how2compress/pretrained/train/exp1709-1/1-0.5740472400763013+-0.007589454466788492-0.982-0.931.pth"
# ours_model = "/how2compress/pretrained/train/exp1710-1/1-0.4620776571761211+-0.0023735605196651965-0.993-0.971.pth"
# ours_model = "/how2compress/pretrained/train/exp1711-1-1/1-0.5398975733473949+0.0023943903714293002-0.991-0.947.pth"
# ours_model = "/how2compress/pretrained/train/exp1713-1-1/1-0.3470107100380082+-0.01633170276732193-0.991-0.952.pth"
ours_dest_path = f"test_bitrate_chunk-ours-{dataset}.h264"

uniqp_dest_path = f"test_bitrate_chunk-uniqp-{dataset}.h264"
spatial_aq_dest_path = f"test_bitrate_chunk-spatial_aq-{dataset}.h264"

num = 30
start_idx = 1

DEVICE = "cuda"
RESIZE_FACTOR = 4
BATCH_SIZE = 1

# mb_w, mb_h = cals.macroblocks_wh(1920, 1080)
# transform = image_ops.vit_transform_fn()
# resizer = image_ops.resize_img_tensor((mb_h * 4, mb_w * 4))

subprocess.run(
    [
        "/myh264/bin/ffmpeg",
        "-y",
        "-i",
        f"/how2compress/data/MOT17Det/train/{dataset}/img1/%06d.jpg",
        "-start_number",
        str(start_idx),
        "-vframes",
        str(num),
        "-framerate",
        "30",
        "-qp",
        "10",
        "-pix_fmt",
        "yuv420p",
        os.path.join(root_dir, f"video-uni30-{dataset}.mp4"),
    ]
)

# # accmpeg
# model = AccMpeg(mb_h, mb_w).to(DEVICE)
# model.load_state_dict(torch.load(accmpeg_model))
# count = 0
# with open("/myh264/qp_matrix_file", "w") as f:
#     prev_matrix = []
#     for i in range(num):
#         count += 1
#         image = cv2.imread(os.path.join(raw, f"{(i+start_idx):06d}.jpg"))
#         image = image_ops.wrap_img(image)
#         image = transform(image).unsqueeze(0).to(DEVICE)
#         # print(image.shape)
#         resize_image = resizer(image)
#         ems_map_indices, ems_map_v, selections = model(resize_image)
#         selections = [mapping[i] for _, i in selections[0]]
#         # print(selections)
#         if i % 5 == 0:
#             matrix = np.reshape(selections, (mb_h, mb_w))
#             prev_matrix = matrix
#             # print(matrix)
#             for row in matrix:
#                 f.write(" ".join(map(str, row)) + "\n")
#         else:
#             for row in prev_matrix:
#                 f.write(" ".join(map(str, row)) + "\n")

# subprocess.run(
#     [
#         "/myh264/bin/ffmpeg",
#         "-y",
#         "-i",
#         f"/how2compress/data/MOT17Det/train/{dataset}/img1/%06d.jpg",
#         "-start_number",
#         str(start_idx),
#         "-vframes",
#         str(num),
#         "-framerate",
#         "30",
#         "-qp",
#         "10",
#         "-pix_fmt",
#         "yuv420p",
#         os.path.join(root_dir, f"video-accmpeg-{dataset}.mp4"),
#     ]
# )

# # ours
# model = MobileVitV2()
# model.load_state_dict(torch.load(ours_model))
# model.to(DEVICE)
# model.set_output_size((mb_h, mb_w))

# count = 0
# with open("/myh264/qp_matrix_file", "w") as f:
#     prev_matrix = []
#     for i in range(num):
#         # count += 1
#         image = cv2.imread(os.path.join(raw, f"{(i+start_idx):06d}.jpg"))
#         image = image_ops.wrap_img(image)
#         image = transform(image).unsqueeze(0).to(DEVICE)
#         # print(image.shape)
#         resize_image = resizer(image)
#         ems_map_indices, ems_map_v, selections = model(resize_image)
#         # print(selections[0])
#         selections = [
#             mapping[max(0, i)] if i == 4 else mapping[i] for _, i in selections[0]
#         ]
#         # print(matrix)
#         # print(selections)
#         if i % 5 == 0:
#             matrix = np.reshape(selections, (mb_h, mb_w))
#             prev_matrix = matrix
#             for row in matrix:
#                 f.write(" ".join(map(str, row)) + "\n")
#         else:
#             for row in prev_matrix:
#                 f.write(" ".join(map(str, row)) + "\n")


# subprocess.run(
#     [
#         "/myh264/bin/ffmpeg",
#         "-y",
#         "-i",
#         f"/how2compress/data/MOT17Det/train/{dataset}/img1/%06d.jpg",
#         "-start_number",
#         str(start_idx),
#         "-vframes",
#         str(num),
#         "-framerate",
#         "30",
#         "-qp",
#         "10",
#         "-pix_fmt",
#         "yuv420p",
#         os.path.join(root_dir, f"video-ours-{dataset}.mp4"),
#     ]
# )


# exe = "/how2compress/src/tools/AppEncCudaEM"
# exe1 = "/how2compress/src/tools/AppEncCudaNoEM"

# # uni qp
# result1 = subprocess.run(
#     [
#         exe1,
#         "-i",
#         video,
#         "-o",
#         os.path.join(root_dir, uniqp_dest_path),
#         "-s",
#         "1920x1080",
#         "-gpu",
#         "0",
#         "-qmin",
#         "30",
#         "-gop",
#         "30",
#         "-qmax",
#         "30",
#         "-constqp",
#         "30",
#         "-initqp",
#         "30",
#         "-tuninginfo",
#         "ultralowlatency",
#         "-rc",
#         "constqp",
#     ]
# )

# # spatial AQ
# subprocess.run(
#     [
#         exe1,
#         "-i",
#         video,
#         "-o",
#         os.path.join(root_dir, spatial_aq_dest_path),
#         "-s",
#         "1920x1080",
#         "-gop",
#         "30",
#         "-qmin",
#         "30",
#         "-qmax",
#         "45",
#         "-aq",
#         "0",
#         "-initqp",
#         "35",
#         "-gpu",
#         "0",
#         "-tuninginfo",
#         "ultralowlatency",
#         "-rc",
#         "cbr",
#     ],
#     stdout=subprocess.PIPE,
#     stderr=subprocess.PIPE,
#     check=True,
# )

# # accmpeg
# model = AccMpeg(mb_h, mb_w).to(DEVICE)
# model.load_state_dict(torch.load(accmpeg_model))
# count = 0
# with open("test_bitrate_emphasis.txt", "w") as f:
#     for i in range(30):
#         count += 1
#         image = cv2.imread(os.path.join(raw, f"{(i+start_idx):06d}.jpg"))
#         image = image_ops.wrap_img(image)
#         image = transform(image).unsqueeze(0).to(DEVICE)
#         # print(image.shape)
#         resize_image = resizer(image)
#         ems_map_indices, ems_map_v, selections = model(resize_image)
#         f.write(",".join([str(level) for _, level in selections[0]]) + "\n")

# result = subprocess.run(
#     [
#         exe,
#         "-i",
#         video,
#         "-o",
#         os.path.join(root_dir, accmpeg_dest_path),
#         "-s",
#         "1920x1080",
#         "-gpu",
#         "0",
#         "-e",
#         "test_bitrate_emphasis.txt",
#         "-gop",
#         "30",
#         "-qmin",
#         "30",
#         "-qmax",
#         "45",
#         "-constqp",
#         "45",
#         "-initqp",
#         "45",
#         "-tuninginfo",
#         "ultralowlatency",
#         "-rc",
#         "constqp",
#     ]
# )

# # ours
# model = MobileVitV2()
# model.load_state_dict(torch.load(ours_model))
# model.to(DEVICE)
# model.set_output_size((mb_h, mb_w))
# count = 0
# with open("test_bitrate_emphasis.txt", "w") as f:
#     for i in range(num):
#         count += 1
#         image = cv2.imread(os.path.join(raw, f"{(i+start_idx):06d}.jpg"))
#         image = image_ops.wrap_img(image)
#         image = transform(image).unsqueeze(0).to(DEVICE)
#         # print(image.shape)
#         resize_image = resizer(image)
#         ems_map_indices, ems_map_v, selections = model(resize_image)
#         f.write(",".join([str(level) for _, level in selections[0]]) + "\n")

# result = subprocess.run(
#     [
#         exe,
#         "-i",
#         video,
#         "-o",
#         os.path.join(root_dir, ours_dest_path),
#         "-s",
#         "1920x1080",
#         "-gpu",
#         "0",
#         "-e",
#         "test_bitrate_emphasis.txt",
#         "-gop",
#         "30",
#         "-qmin",
#         "30",
#         "-qmax",
#         "45",
#         "-constqp",
#         "45",
#         "-initqp",
#         "45",
#         "-tuninginfo",
#         "ultralowlatency",
#         "-rc",
#         "constqp",
#     ]
# )
