import torch
import cv2
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model.hgmodel import MobileVitV2
from src.model.utils import ems2selections
from src.utils import image_ops, load, cals, metrics
from src.dataset.dataloader import PandaDataset, collate_fn
import os
from ultralytics import YOLO
from tqdm import tqdm
import supervision as sv
import time

QP_MAPPING = {45: 0, 41: 1, 37: 2, 34: 3, 30: 4}
DEVICE = "cuda:0"
RESIZE_FACTOR = 4
DATASETS = [
    "01_University_Canteen",
    # "02_OCT_Habour",
    # "03_Xili_Crossroad",
    # "04_Primary_School",
    # "05_Basketball_Court",
    # "06_Xinzhongguan",
    # "07_University_Campus",
    # "08_Xili_Street_1",
    # "09_Xili_Street_2",
    # "10_Huaqiangbei",
]
BATCH_SIZE = 1


model = MobileVitV2()
model.load_state_dict(
    torch.load(
        # "/how2compress/pretrained/train/exp-ddp-30-45-mot17-04/0-best.pth"
        # "/how2compress/pretrained/train/exp-ddp-30-45-all/0-best.pth"
        # "/how2compress/pretrained/train/exp-ddp-30-45-1709/0-best.pth"
        # "/how2compress/pretrained/train/exp-ddp-30-45-1710/2-best.pth"
        "/how2compress/pretrained/train/exp-ddp-30-45-panda-01-12/0-best.pth"
    )
)
model.to(DEVICE)

inferencer = YOLO(
    "/how2compress/pretrained/best-panda-2560.pt", verbose=False
).to(DEVICE)

transform = image_ops.vit_transform_fn()
root = "/how2compress/data/pandasH264"
# dataset = "PANDA"
results_root = "/how2compress/results"
r = os.path.join(results_root, "eval30-45-panda-01-12-0.txt")
rd = os.path.join(results_root, "decisions30-45-panda-01-12-0.txt")
# path = os.path.join(root, dataset, "30")
# frames = sorted(os.listdir(path))
# frames = [os.path.join(path, frame) for frame in frames]
# enc_frames_dir = os.path.join(results_root, dataset)
# if not os.path.exists(enc_frames_dir):
#     os.makedirs(enc_frames_dir)

# load gt
dataset = PandaDataset(
    dataset_dir="/how2compress/data/pandasRS",
    gt_dir="/how2compress/data/pandas/unzipped/train_annos",
    reference_dir="/how2compress/data/pandasH264",
    yuv_dir="/how2compress/data/pandasYUV",
    resize_factor=RESIZE_FACTOR,
)

model.eval()

mAPs = []
mAPs_gt = []
frames_size = []
times = []
ret_frames = []
decisions = []
for seq in DATASETS:
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

    for images, labels, indices in tqdm(dataloader, desc=f"{seq} val"):
        images = images.to(DEVICE)
        resize_images = resizer(images)
        # print(images.shape)
        with torch.no_grad():
            start_time = time.time()
            ems_map_indices, ems_map_v, selections = model(resizer(images))
            end_time = time.time()
            times.append(end_time - start_time)

        # noisy adjustment
        # print(ems_map_indices.shape)
        # print(len(selections[0]))
        # adjust_indices = model.noisy_adjustment(ems_map_indices, mb_h, mb_w, 0.99)
        # selections = ems2selections(adjust_indices.cpu().numpy())

        targets = inferencer.predict(images, classes=[1], verbose=False)
        targets = metrics.yolo2sv(targets)
        targets = [
            metrics.normalize_detections(
                det,
                (
                    images.shape[3],
                    images.shape[2],
                ),
            )
            for det in targets
        ]
        ret_selections = [[level for _, level in selection] for selection in selections]
        decisions.extend(ret_selections)
        compressed_images, sizes, enc_frames = dataset.enc_and_ret_val(
            indices, selections, DEVICE
        )
        ret_frames.extend(enc_frames)
        compressed_images = compressed_images.to(DEVICE)
        preds = inferencer.predict(compressed_images, classes=[1], verbose=False)
        preds = metrics.yolo2sv(preds)
        preds = [
            metrics.normalize_detections(
                det,
                (
                    dataset.curr_seq_property["width"],
                    dataset.curr_seq_property["height"],
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
