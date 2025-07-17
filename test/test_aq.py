import torch
import cv2
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model.hgmodel import MobileVitV2
from src.model.utils import ems2selections
from src.utils import image_ops, load, cals, metrics
from src.dataset.dataloader import MOTDataset, collate_fn
import os
from ultralytics import YOLO
from tqdm import tqdm
import supervision as sv
import time
import argparse

DEVICE = "cuda:1"
RESIZE_FACTOR = 4
# DATASET = [args.dataset]
BATCH_SIZE = 1

# DATASET = ["MOT17-10", "MOT17-11", "MOT17-13"]
DATASET = ["MOT17-13"]

inferencer = YOLO("/how2compress/pretrained/mot17-m.pt", verbose=False).to(
    DEVICE
)
# load gt
dataset = MOTDataset(
    dataset_dir="/how2compress/data/MOT17Det/train",
    reference_dir="/how2compress/data/detections",
    # ssim_label_dir="/how2compress/data/ssim_labels",
    yuv_dir="/how2compress/data/MOT17DetYUV",
    resize_factor=RESIZE_FACTOR,
)

num = 25

result_file = "acc-result.txt"


def load_video(filepath: str):
    cap = cv2.VideoCapture(filepath)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


for seq in DATASET:
    dataset.load_sequence(seq)
    aq_root = f"/how2compress/data/UNI30CHUNK/{seq}"
    ours_root = f"/how2compress/video-result/{seq}"
    accmpeg_root = f"/how2compress/video-result/{seq}"
    uni_root = f"/how2compress/data/UNI30CHUNK/{seq}"
    aq_chunks = sorted([f for f in os.listdir(aq_root) if f.startswith("aq")])
    ours_chunks = sorted(
        [f for f in os.listdir(ours_root) if f.startswith("h264-ours")]
    )
    accmpeg_chunks = sorted(
        [f for f in os.listdir(accmpeg_root) if f.startswith("h264-accmpeg")]
    )
    uni_chunks = sorted([f for f in os.listdir(uni_root) if f.startswith("uni")])

    # assert (
    #     len(aq_chunks) == len(ours_chunks) == len(accmpeg_chunks) == len(uni_chunks)
    # ), f"len(aq_chunks)={len(aq_chunks)}, len(ours_chunks)={len(ours_chunks)}, len(accmpeg_chunks)={len(accmpeg_chunks)}, len(uni_chunks)={len(uni_chunks)}"
    mAPs_uni = []
    mAPs_aq = []
    mAPs_accmpeg = []
    mAPs_ours = []

    count = 0
    for uni_chunk, aq_chunk, accmpeg_chunk, ours_chunk in zip(
        uni_chunks, aq_chunks, accmpeg_chunks, ours_chunks
    ):
        uni_frames = load_video(os.path.join(uni_root, uni_chunk))
        aq_frames = load_video(os.path.join(aq_root, aq_chunk))
        accmpeg_frames = load_video(os.path.join(accmpeg_root, accmpeg_chunk))
        ours_frames = load_video(os.path.join(ours_root, ours_chunk))
        assert (
            len(uni_frames) == len(aq_frames) == len(accmpeg_frames) == len(ours_frames)
        ), f"len(uni_frames)={len(uni_frames)}, len(aq_frames)={len(aq_frames)}, len(accmpeg_frames)={len(accmpeg_frames)}, len(ours_frames)={len(ours_frames)}"

        for uni_frame, aq_frame, accmpeg_frame, ours_frame in tqdm(
            zip(uni_frames, aq_frames, accmpeg_frames, ours_frames),
            desc=f"{seq}-{count}",
        ):
            labels = [dataset.curr_labels[count]]
            uni_frame = image_ops.wrap_img(uni_frame)
            aq_frame = image_ops.wrap_img(aq_frame)
            accmpeg_frame = image_ops.wrap_img(accmpeg_frame)
            ours_frame = image_ops.wrap_img(ours_frame)
            uni_frame = image_ops.vit_transform_fn()(uni_frame)
            aq_frame = image_ops.vit_transform_fn()(aq_frame)
            accmpeg_frame = image_ops.vit_transform_fn()(accmpeg_frame)
            ours_frame = image_ops.vit_transform_fn()(ours_frame)
            uni_frame = uni_frame.unsqueeze(0).to(DEVICE)
            aq_frame = aq_frame.unsqueeze(0).to(DEVICE)
            accmpeg_frame = accmpeg_frame.unsqueeze(0).to(DEVICE)
            ours_frame = ours_frame.unsqueeze(0).to(DEVICE)

            uni_targets = inferencer.predict(uni_frame, classes=[1], verbose=False)
            uni_targets = metrics.yolo2sv(uni_targets)
            uni_targets = [
                metrics.normalize_detections(
                    det,
                    (
                        dataset.curr_seq_property["width"],
                        dataset.curr_seq_property["height"],
                    ),
                )
                for det in uni_targets
            ]

            aq_targets = inferencer.predict(aq_frame, classes=[1], verbose=False)
            aq_targets = metrics.yolo2sv(aq_targets)
            aq_targets = [
                metrics.normalize_detections(
                    det,
                    (
                        dataset.curr_seq_property["width"],
                        dataset.curr_seq_property["height"],
                    ),
                )
                for det in aq_targets
            ]

            accmpeg_targets = inferencer.predict(
                accmpeg_frame, classes=[1], verbose=False
            )
            accmpeg_targets = metrics.yolo2sv(accmpeg_targets)
            accmpeg_targets = [
                metrics.normalize_detections(
                    det,
                    (
                        dataset.curr_seq_property["width"],
                        dataset.curr_seq_property["height"],
                    ),
                )
                for det in accmpeg_targets
            ]

            ours_targets = inferencer.predict(ours_frame, classes=[1], verbose=False)
            ours_targets = metrics.yolo2sv(ours_targets)
            ours_targets = [
                metrics.normalize_detections(
                    det,
                    (
                        dataset.curr_seq_property["width"],
                        dataset.curr_seq_property["height"],
                    ),
                )
                for det in ours_targets
            ]

            mAP_uni = sv.MeanAveragePrecision.from_detections(uni_targets, labels)
            mAP_aq = sv.MeanAveragePrecision.from_detections(aq_targets, labels)
            mAP_accmpeg = sv.MeanAveragePrecision.from_detections(
                accmpeg_targets, labels
            )
            mAP_ours = sv.MeanAveragePrecision.from_detections(ours_targets, labels)

            mAPs_uni.append(mAP_uni)
            mAPs_aq.append(mAP_aq)
            mAPs_accmpeg.append(mAP_accmpeg)
            mAPs_ours.append(mAP_ours)
            count += 1

            print(
                mAP_uni.map50_95,
                mAP_aq.map50_95,
                mAP_accmpeg.map50_95,
                mAP_ours.map50_95,
            )

    # append the result to file
    with open(result_file, "a") as f:
        mean_map50_95_uni = sum([mAP.map50_95 for mAP in mAPs_uni]) / len(mAPs_uni)
        mean_map75_uni = sum([mAP.map75 for mAP in mAPs_uni]) / len(mAPs_uni)
        mean_map50_uni = sum([mAP.map50 for mAP in mAPs_uni]) / len(mAPs_uni)
        mean_map50_95_aq = sum([mAP.map50_95 for mAP in mAPs_aq]) / len(mAPs_aq)
        mean_map75_aq = sum([mAP.map75 for mAP in mAPs_aq]) / len(mAPs_aq)
        mean_map50_aq = sum([mAP.map50 for mAP in mAPs_aq]) / len(mAPs_aq)
        mean_map50_95_accmpeg = sum([mAP.map50_95 for mAP in mAPs_accmpeg]) / len(
            mAPs_accmpeg
        )
        mean_map75_accmpeg = sum([mAP.map75 for mAP in mAPs_accmpeg]) / len(
            mAPs_accmpeg
        )
        mean_map50_accmpeg = sum([mAP.map50 for mAP in mAPs_accmpeg]) / len(
            mAPs_accmpeg
        )
        mean_map50_95_ours = sum([mAP.map50_95 for mAP in mAPs_ours]) / len(mAPs_ours)
        mean_map75_ours = sum([mAP.map75 for mAP in mAPs_ours]) / len(mAPs_ours)
        mean_map50_ours = sum([mAP.map50 for mAP in mAPs_ours]) / len(mAPs_ours)
        f.write(
            f"{seq},{mean_map50_95_uni},{mean_map75_uni},{mean_map50_uni},{mean_map50_95_aq},{mean_map75_aq},{mean_map50_aq},{mean_map50_95_accmpeg},{mean_map75_accmpeg},{mean_map50_accmpeg},{mean_map50_95_ours},{mean_map75_ours},{mean_map50_ours}\n"
        )
