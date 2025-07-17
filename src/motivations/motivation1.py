"""motivation 1

Encode the object region with highest quality and 'background' region with lower quality. We compare the results
with the performance with all frames are encoded with the same high quality.

The goal for this is to evaluate:
(1) the background region is also important for the object detection task,
"""

import supervision as sv
from typing import List
import torch
from src.model.hgmodel import MobileVitV2
from src.model.utils import ems2selections
from src.utils import image_ops, load, cals, metrics
from src.dataset.dataloader import MOTDataset, collate_fn
from torch.utils.data import DataLoader

from ultralytics import YOLO
from tqdm import tqdm
import os

DEVICE = "cuda:0"
RESIZE_FACTOR = 4
BATCH_SIZE = 1


def get_ROI_mb_indices(
    labels: List[sv.Detections],
    mb_h: int,
    mb_w: int,
) -> List[List[int]]:
    scale_factor = torch.tensor([mb_w, mb_h, mb_w, mb_h])

    # indices in/outside the bbox
    masks = []

    # group mb indices with in/outside the bbox
    for label in labels:
        xyxy = label.xyxy  # (N, 4)
        xyxy = torch.from_numpy(xyxy)
        xyxy = torch.ceil(xyxy * scale_factor).int()
        # filter the raster order of index which is located in the bbox
        mask = torch.zeros((mb_h, mb_w), dtype=torch.int)
        for bbox in xyxy:
            x1, y1, x2, y2 = bbox
            mask[y1:y2, x1:x2] = 4  # assign the highest quality to ROI area
        mask = mask.view(-1)
        masks.append(mask)
    masks = torch.stack(masks).numpy().tolist()
    masks = [[(i, v) for i, v in enumerate(mask)] for mask in masks]
    return masks


def get_ROI_mb_indices_reverse(
    labels: List[sv.Detections],
    mb_h: int,
    mb_w: int,
) -> List[List[int]]:
    scale_factor = torch.tensor([mb_w, mb_h, mb_w, mb_h])

    # indices in/outside the bbox
    masks = []

    # group mb indices with in/outside the bbox
    for label in labels:
        xyxy = label.xyxy  # (N, 4)
        xyxy = torch.from_numpy(xyxy)
        xyxy = torch.ceil(xyxy * scale_factor).int()
        # filter the raster order of index which is located in the bbox
        mask = torch.ones((mb_h, mb_w), dtype=torch.int) * 4
        for bbox in xyxy:
            x1, y1, x2, y2 = bbox
            mask[y1:y2, x1:x2] = 0  # assign the highest quality to ROI area
        mask = mask.view(-1)
        masks.append(mask)
    masks = torch.stack(masks).numpy().tolist()
    masks = [[(i, v) for i, v in enumerate(mask)] for mask in masks]
    return masks


def run(seq: str, qmin: int, qmax: int, reverse: bool = False):
    results_root = "/how2compress/results"
    r = os.path.join(
        results_root,
        f"motivation1-{seq}-{qmin}-{qmax}-{'rev' if reverse else ''}-2.txt",
    )
    # rd = os.path.join(results_root, "decisions30-45-mot17-02-f.txt")
    mAPs = []
    mAPs_gt = []
    frames_size = []
    inferencer = YOLO(
        "/how2compress/pretrained/best_MOT_1920.pt", verbose=False
    ).to(DEVICE)
    # transform = image_ops.vit_transform_fn()
    dataset = MOTDataset(
        dataset_dir="/how2compress/data/MOT17Det/train",
        reference_dir="/how2compress/data/detections",
        ssim_label_dir="/how2compress/data/ssim_labels",
        yuv_dir="/how2compress/data/MOT17DetYUV",
        resize_factor=RESIZE_FACTOR,
    )

    dataset.load_sequence(seq)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False
    )
    height, width = (
        dataset.curr_seq_property["height"],
        dataset.curr_seq_property["width"],
    )
    mb_w, mb_h = cals.macroblocks_wh(width, height)
    # resizer = image_ops.resize_img_tensor((mb_h * 4, mb_w * 4))

    for images, labels, indices in tqdm(dataloader):
        images = images.to(DEVICE)
        targets = inferencer.predict(images, classes=[1], verbose=False)
        targets = metrics.yolo2sv(targets)
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
        if reverse:
            selections = get_ROI_mb_indices_reverse(targets, mb_h, mb_w)
        else:
            selections = get_ROI_mb_indices(targets, mb_h, mb_w)
        compressed_images, sizes, enc_frames = dataset.enc_and_ret_val(
            indices, selections, DEVICE, qmin, qmax
        )

        compressed_images = compressed_images.to(DEVICE)
        preds = inferencer.predict(compressed_images, classes=[1], verbose=False)
        preds = metrics.yolo2sv(preds)
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
        # assert len(preds) == len(
        #     targets
        # ), f"preds size {len(preds)} != targets size {len(targets)}"

        frames_size.extend(sizes)
        mAP_t = sv.MeanAveragePrecision.from_detections(targets, labels)
        mAP = sv.MeanAveragePrecision.from_detections(preds, labels)
        mAPs.append(mAP)
        mAPs_gt.append(mAP_t)

    with open(r, "a") as f:
        for mAP, mAP_gt, frame_size in zip(mAPs, mAPs_gt, frames_size):
            f.write(
                f"{mAP.map50_95},{mAP.map75},{mAP.map50},{mAP_gt.map50_95},{mAP_gt.map75},{mAP_gt.map50},{frame_size},0\n"
            )


if __name__ == "__main__":
    run()
