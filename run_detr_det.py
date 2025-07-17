import torch
import cv2
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model.hgmodel import MobileVitV2
from src.model.utils import ems2selections
from src.utils import image_ops, load, cals, metrics
from src.dataset.dataloader import MOTDataset, collate_fn
from src.inferencer.detr import DETRInferencer
import os
from ultralytics import YOLO
from tqdm import tqdm
import supervision as sv
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="dataset name")
parser.add_argument("--model", type=str, help="model path")
# parser.add_argument("--results", type=str, help="results path")

args = parser.parse_args()

DEVICE = "cuda:1"
RESIZE_FACTOR = 4
DATASET = [args.dataset]
BATCH_SIZE = 1


model = MobileVitV2()
model.load_state_dict(
    torch.load(
        # "/how2compress/pretrained/train/exp-ddp-30-45-mot17-04/0-best.pth"
        # "/how2compress/pretrained/train/exp-ddp-30-45-all/0-best.pth"
        # "/how2compress/pretrained/train/exp-ddp-30-45-1709/0-best.pth"
        # "/how2compress/pretrained/train/exp-ddp-30-45-1710/2-best.pth"
        args.model
    )
)
model.to(DEVICE)

# inferencer = YOLO("/how2compress/pretrained/mot17-m.pt", verbose=False).to(
#     DEVICE
# )
inferencer = DETRInferencer("50")
inferencer.model.to(DEVICE)
print(f"inferencer model on: {inferencer.model.device}")

transform = image_ops.vit_transform_fn()
root = "/how2compress/data/MOT17DetH264"
dataset = args.dataset
results_root = "/how2compress/results"
r = os.path.join(results_root, f"eval30-45-{dataset}-detr.txt")
rd = os.path.join(results_root, f"decisions30-45-{dataset}-detr.txt")
path = os.path.join(root, dataset, "30")
frames = sorted(os.listdir(path))
frames = [os.path.join(path, frame) for frame in frames]
enc_frames_dir = os.path.join(results_root, dataset)
if not os.path.exists(enc_frames_dir):
    os.makedirs(enc_frames_dir)

# load gt
dataset = MOTDataset(
    dataset_dir="/how2compress/data/MOT17Det/train",
    reference_dir="/how2compress/data/detections",
    ssim_label_dir="/how2compress/data/ssim_labels",
    yuv_dir="/how2compress/data/MOT17DetYUV",
    resize_factor=RESIZE_FACTOR,
)

model.eval()

mAPs = []
mAPs_gt = []
frames_size = []
times = []
ret_frames = []
decisions = []
for seq in DATASET:
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
    model.set_output_size((mb_h, mb_w))
    count = 0
    for images, labels, indices in tqdm(dataloader, desc=f"{seq} val"):
        count += 1
        images = images.to(DEVICE)
        resize_images = resizer(images)
        start_time = time.time()
        ems_map_indices, ems_map_v, selections = model(resize_images)
        end_time = time.time()
        times.append(end_time - start_time)
        # assert images.shape == (1, 3, 1080, 1920), f"images shape: {images.shape}"
        targets = inferencer.predict(images, imgsz=(1080, 1920))
        # targets = metrics.yolo2sv(targets)
        targets = [
            metrics.normalize_detections(
                det,
                (
                    dataset.curr_seq_property["width"],
                    dataset.curr_seq_property["height"],
                    # images.shape[3],
                    # images.shape[2],
                ),
            )
            for det in targets
        ]
        ret_selections = [[level for _, level in selection] for selection in selections]
        decisions.extend(ret_selections)
        compressed_images, sizes, enc_frames = dataset.enc_and_ret_val(
            indices, selections, DEVICE
        )
        # cv2.imwrite(f"{count:06d}.png", enc_frames[0])
        ret_frames.extend(enc_frames)
        compressed_images = compressed_images.to(DEVICE)
        preds = inferencer.predict(compressed_images, imgsz=(1080, 1920))
        # preds = metrics.yolo2sv(preds)
        preds = [
            metrics.normalize_detections(
                det,
                (
                    dataset.curr_seq_property["width"],
                    dataset.curr_seq_property["height"],
                    # images.shape[3],
                    # images.shape[2],
                ),
            )
            for det in preds
        ]
        assert len(preds) == len(
            targets
        ), f"preds size {len(preds)} != targets size {len(targets)}"

        frames_size.extend(sizes)
        mAP_t = sv.MeanAveragePrecision.from_detections(targets, labels)
        mAP = sv.MeanAveragePrecision.from_detections(preds, labels)
        mAPs.append(mAP)
        mAPs_gt.append(mAP_t)

with open(r, "a") as f:
    for mAP, mAP_gt, frame_size, t in zip(mAPs, mAPs_gt, frames_size, times):
        f.write(
            f"{mAP.map50_95},{mAP.map75},{mAP.map50},{mAP_gt.map50_95},{mAP_gt.map75},{mAP_gt.map50},{frame_size},{t}\n"
        )

with open(rd, "a") as f:
    for decision in decisions:
        f.write(",".join(map(str, decision)) + "\n")

# for i, frame in enumerate(ret_frames):
#     cv2.imwrite(os.path.join(enc_frames_dir, f"{i+1:06d}.png"), frame)
