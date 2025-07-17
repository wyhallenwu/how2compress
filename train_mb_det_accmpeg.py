import numpy as np
import supervision as sv
import torch
from datetime import datetime
from src.utils.image_ops import resize_img_tensor
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
from ultralytics import YOLO

import src.utils.metrics as metrics
from src.model.utils import ems2selections
import wandb
from src.dataset.dataloader import MOTDataset, collate_fn
from src.model.hgmodel import MobileVitV2
from src.model.am_model import AccMpeg
from src.utils.cals import macroblocks_wh
from src.utils.cals import mb_ssim
import time
import os
import csv
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import argparse

from torch.utils.tensorboard import SummaryWriter


NAME = "exp-ddp-30-45-accmpeg-1704-0.4q-choices5"
DATASETS = [
    "MOT17-02",
    # "MOT17-04",
    # # "MOT17-05",
    # "MOT17-09",
    # "MOT17-10",
    # "MOT17-11",
    # "MOT17-13",
    # "MOT20-02"
]
QP_MAPPING = {45: 0, 41: 1, 37: 2, 34: 3, 30: 4}

QP_SET = [30, 34, 37, 41, 45]
EPOCH = 2
LR = 1e-3
MIN_LR = 1e-6
BATCH_SIZE = 14
TRAIN_RATIO = 0.7
REWARD_MAPPING = [1, 1.3, 1.6, 1.9, 2.2]
WEIGHT_MAP = [10]
RESIZE_FACTOR = 4
THRESHOLD_INSIDE = 0.97
THRESHOLD_OUTSIDE = 0.95
EXPLORE_PROB = 0.8
EXPLORE_DECAY = 0.1
CHOICES = 5


def train_epoch(
    model: AccMpeg,
    inferencer: YOLO,
    optimizer: torch.optim.Optimizer,
    dataset: MOTDataset,
    device_id: int,
    rank: int,
    world_size: int,
    curr_epoch: int,
):
    train_detections = {}
    val_detections = {}
    train_selections = {}
    val_selections = {}

    train_mean_mAP50_95 = {}
    train_mean_mAP75 = {}
    train_mean_mAP50 = {}

    val_mean_mAP50_95 = {}
    val_mean_mAP75 = {}
    val_mean_mAP50 = {}
    val_mean_gt_mAP50_95 = {}
    val_mean_gt_mAP75 = {}
    val_mean_gt_mAP50 = {}

    train_loss = {}
    train_loss1 = {}
    train_loss2 = {}

    val_loss = {}
    val_loss1 = {}
    val_loss2 = {}

    train_compressed_images_size = {}
    val_compressed_images_size = {}

    explore_probs = {}

    for seq in DATASETS:
        labels = model.load_labels(seq)
        # split into train and val
        dataset.load_sequence(seq)
        train_size = int(TRAIN_RATIO * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, test_size])

        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=collate_fn,
            sampler=train_sampler,
            num_workers=8,
            # pin_memory=True,
            drop_last=False,
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=collate_fn,
            sampler=val_sampler,
            # pin_memory=True,
            drop_last=False,
        )
        train_sampler.set_epoch(curr_epoch)
        val_sampler.set_epoch(curr_epoch)

        height, width = (
            dataset.curr_seq_property["height"],
            dataset.curr_seq_property["width"],
        )
        # config
        # reward_mapping = torch.exp(torch.tensor(REWARD_MAPPING)).to(device_id)
        # weight_mAP = torch.tensor(WEIGHT_MAP).to(device_id)

        # init log
        train_loss.setdefault(seq, [])
        train_loss1.setdefault(seq, [])
        train_loss2.setdefault(seq, [])
        val_loss.setdefault(seq, [])
        val_loss1.setdefault(seq, [])
        val_loss2.setdefault(seq, [])
        train_mean_mAP50_95.setdefault(seq, [])
        train_mean_mAP75.setdefault(seq, [])
        train_mean_mAP50.setdefault(seq, [])
        val_mean_mAP50_95.setdefault(seq, [])
        val_mean_mAP75.setdefault(seq, [])
        val_mean_mAP50.setdefault(seq, [])
        val_mean_gt_mAP50_95.setdefault(seq, [])
        val_mean_gt_mAP75.setdefault(seq, [])
        val_mean_gt_mAP50.setdefault(seq, [])
        train_detections.setdefault(seq, [[], []])
        val_detections.setdefault(seq, [[], []])
        train_selections.setdefault(seq, [])
        val_selections.setdefault(seq, [])
        train_compressed_images_size.setdefault(seq, [])
        val_compressed_images_size.setdefault(seq, [])

        explore_probs.setdefault(seq, [])

        model.train()
        step = 0
        mb_w, mb_h = macroblocks_wh(width, height)
        resizer = resize_img_tensor((mb_h * 4, mb_w * 4))
        # model.set_output_size((mb_h, mb_w))

        for images, labels, indices in tqdm(
            train_dataloader, desc=f"[TRAIN] EPOCH: {curr_epoch+1}, SEQ: {seq}"
        ):
            images = images.to(device_id)
            ems_map, ems_map_v, selections = model(resizer(images))

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

            compressed_images, _ = dataset.enc_and_ret(indices, selections, device_id)
            compressed_images = compressed_images.to(device_id)

            # ssim_diffs, _ = mb_ssim(compressed_images, images)
            # adjust_indices = model.get_adjusted_labels(
            #     labels,
            #     ems_map,
            #     ssim_diffs,
            #     mb_h,
            #     mb_w,
            #     max(EXPLORE_PROB - EXPLORE_DECAY * step, 0.1),
            # )
            # selections = ems2selections(adjust_indices.cpu().numpy())
            compressed_images, compressed_images_size = dataset.enc_and_ret(
                indices, selections, device_id
            )
            compressed_images = compressed_images.to(device_id)

            preds = inferencer.predict(compressed_images, classes=[1], verbose=False)
            preds = metrics.yolo2sv(preds)
            preds = [
                metrics.normalize_detections(
                    det,
                    (
                        dataset.curr_seq_property["width"],
                        dataset.curr_seq_property["height"],
                        # compressed_images.shape[3],
                        # compressed_images.shape[2],
                    ),
                )
                for det in preds
            ]
            mAP = sv.MeanAveragePrecision.from_detections(preds, labels)
            acc_target = model.label[indices, :].to(device_id)
            # print(f"type of acc_target: {type(acc_target)}")
            # print(f"acc_target: {acc_target.shape}")
            # print(f"type of ems_map_v: {type(ems_map_v)}")
            # print(f"ems_map_v: {ems_map_v.shape}")

            loss = model.loss_fn_det(ems_map_v, acc_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            # log
            train_loss[seq].append(loss.item())
            train_mean_mAP50_95[seq].append(mAP.map50_95)
            train_mean_mAP75[seq].append(mAP.map75)
            train_mean_mAP50[seq].append(mAP.map50)
            train_detections[seq][0].extend(preds)
            train_detections[seq][1].extend(labels)
            train_selections[seq].extend(selections)
            train_compressed_images_size[seq].extend(compressed_images_size)

        # val
        model.eval()
        for images, labels, indices in tqdm(
            val_dataloader, desc=f"[VAL] EPOCH: {curr_epoch + 1}, SEQ: {seq}"
        ):
            images = images.to(device_id)
            with torch.no_grad():
                ems_map, ems_map_v, selections = model(resizer(images))

            targets = inferencer.predict(images, classes=[1], verbose=False)
            targets = metrics.yolo2sv(targets)
            targets = [
                metrics.normalize_detections(
                    det,
                    (
                        dataset.curr_seq_property["width"],
                        dataset.curr_seq_property["height"],
                        # compressed_images.shape[3],
                        # compressed_images.shape[2],
                    ),
                )
                for det in targets
            ]

            # when val the model, use direct predict selections
            compressed_images, _ = dataset.enc_and_ret(indices, selections, device_id)
            compressed_images = compressed_images.to(device_id)

            # ssim_diffs, _ = mb_ssim(compressed_images, images)
            # adjust_indices = model.get_adjusted_labels(
            #     labels, ems_map, ssim_diffs, mb_h, mb_w
            # )
            # selections = ems2selections(adjust_indices.cpu().numpy())
            # compressed_images, compressed_images_size = dataset.enc_and_ret(
            #     indices, selections, device_id
            # )

            preds = inferencer.predict(compressed_images, classes=[1], verbose=False)
            preds = metrics.yolo2sv(preds)
            preds = [
                metrics.normalize_detections(
                    det,
                    (
                        dataset.curr_seq_property["width"],
                        dataset.curr_seq_property["height"],
                        # compressed_images.shape[3],
                        # compressed_images.shape[2],
                    ),
                )
                for det in preds
            ]
            assert len(preds) == len(
                targets
            ), f"preds size {len(preds)} != targets size {len(targets)}"
            mAP = sv.MeanAveragePrecision.from_detections(preds, labels)
            mAP_gt = sv.MeanAveragePrecision.from_detections(targets, labels)

            loss = model.loss_fn_det(ems_map_v, model.label[indices, :].to(device_id))

            # log
            val_loss[seq].append(loss.item())

            val_mean_mAP50_95[seq].append(mAP.map50_95)
            val_mean_mAP75[seq].append(mAP.map75)
            val_mean_mAP50[seq].append(mAP.map50)
            val_mean_gt_mAP50_95[seq].append(mAP_gt.map50_95)
            val_mean_gt_mAP75[seq].append(mAP_gt.map75)
            val_mean_gt_mAP50[seq].append(mAP_gt.map50)
            val_detections[seq][0].extend(preds)
            val_detections[seq][1].extend(labels)
            val_selections[seq].extend(selections)
            val_compressed_images_size[seq].extend(compressed_images_size)

    return (
        train_loss,
        train_loss1,
        train_loss2,
        val_loss,
        val_loss1,
        val_loss2,
        train_mean_mAP50_95,
        train_mean_mAP75,
        train_mean_mAP50,
        val_mean_mAP50_95,
        val_mean_mAP75,
        val_mean_mAP50,
        val_mean_gt_mAP50_95,
        val_mean_gt_mAP75,
        val_mean_gt_mAP50,
        train_detections,
        val_detections,
        train_selections,
        val_selections,
        train_compressed_images_size,
        val_compressed_images_size,
    )


def setup_log():
    # writer
    wandb.login()
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="cvpr2025",
        tags=["MOT", "V1"],
        group="MOT20",
        # # track hyperparameters and run metadata
        config={
            "epoch": EPOCH,
            "lr": LR,
            "min_lr": MIN_LR,
            "qp_set": QP_SET,
            "batch_size": BATCH_SIZE,
            "resize_factor": RESIZE_FACTOR,
            "datasets": DATASETS,
            "weight mAP": WEIGHT_MAP,
            "reward mapping": REWARD_MAPPING,
            "threshold inside": THRESHOLD_INSIDE,
            "threshold outside": THRESHOLD_OUTSIDE,
        },
    )
    return run


if __name__ == "__main__":
    # init distributed
    # run = setup_log()
    parser = argparse.ArgumentParser()
    # parser.add_argument("--name", type=str)
    # parser.add_argument("--dataset", type=str)
    # args = parser.parse_args()
    # NAME = NAME
    # DATASETS = [args.dataset]
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_id = rank % torch.cuda.device_count()
    # model
    model = AccMpeg(h=1088 // 16, w=1920 // 16, choices=CHOICES).to(device_id)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    inferencer = YOLO(
        "/how2compress/pretrained/best_MOT_1920.pt", verbose=False
    ).to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=LR)

    poly_LR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 32, MIN_LR)
    max_map50_95 = -np.inf

    # checkpoint
    folder = "/how2compress/pretrained/train"
    checkpoint = os.path.join(folder, NAME)
    # config = {
    #     "epoch": EPOCH,
    #     "lr": LR,
    #     "min_lr": MIN_LR,
    #     "qp_set": QP_SET,
    #     "batch_size": BATCH_SIZE,
    #     "resize_factor": RESIZE_FACTOR,
    #     "datasets": DATASETS,
    #     "weight mAP": WEIGHT_MAP,
    #     "reward mapping": REWARD_MAPPING,
    #     "threshold inside": THRESHOLD_INSIDE,
    #     "threshold outside": THRESHOLD_OUTSIDE,
    # }

    if rank == 0:
        if not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
            open(os.path.join(checkpoint, "train_log.txt"), "w").close()
            open(os.path.join(checkpoint, "val_log.txt"), "w").close()
            os.makedirs(os.path.join(checkpoint, "logs"))
        writer = SummaryWriter(os.path.join(checkpoint, "logs"))

    dist.barrier()

    dataset = MOTDataset(
        dataset_dir="/how2compress/data/MOT17Det/train",
        reference_dir="/how2compress/data/detections",
        # ssim_label_dir="/how2compress/data/ssim_labels",
        yuv_dir="/how2compress/data/MOT17DetYUV",
        resize_factor=RESIZE_FACTOR,
    )

    for epoch in range(EPOCH):
        (
            train_loss,
            train_loss1,
            train_loss2,
            val_loss,
            val_loss1,
            val_loss2,
            train_mean_mAP50_95,
            train_mean_mAP75,
            train_mean_mAP50,
            val_mean_mAP50_95,
            val_mean_mAP75,
            val_mean_mAP50,
            val_mean_gt_mAP50_95,
            val_mean_gt_mAP75,
            val_mean_gt_mAP50,
            train_detections,
            val_detections,
            train_selections,
            val_selections,
            train_compressed_images_size,
            val_compressed_images_size,
        ) = train_epoch(
            model, inferencer, optimizer, dataset, device_id, rank, world_size, epoch
        )

        poly_LR.step()
        if rank == 0:
            vf = open(os.path.join(checkpoint, "val_log.txt"), "a")
            tf = open(os.path.join(checkpoint, "train_log.txt"), "a")
            train_log_writer = csv.writer(tf)
            val_log_writer = csv.writer(vf)

            for seq in DATASETS:
                train_counts = []
                for selections in train_selections[seq]:
                    for _, level in selections:
                        train_counts.append(level)
                train_counts = np.bincount(train_counts, minlength=5)
                train_counts = train_counts / np.sum(train_counts) * 100
                val_counts = []
                for selections in val_selections[seq]:
                    for _, level in selections:
                        val_counts.append(level)
                val_counts = np.bincount(val_counts, minlength=5)
                val_counts = val_counts / np.sum(val_counts) * 100

                for compress_img_size, selections, mAP50_95, mAP75, mAP50 in zip(
                    train_compressed_images_size[seq],
                    train_selections[seq],
                    train_mean_mAP50_95[seq],
                    train_mean_mAP75[seq],
                    train_mean_mAP50[seq],
                ):
                    selections = [level for _, level in selections]
                    counts = np.bincount(selections, minlength=5)
                    counts = counts / np.sum(counts) * 100
                    log_entry = (
                        "train",
                        epoch + 1,
                        seq,
                        mAP50_95,
                        mAP75,
                        mAP50,
                        compress_img_size,
                        counts,
                        selections,
                    )
                    train_log_writer.writerow(log_entry)
                    tf.flush()
                for (
                    compress_img_size,
                    selections,
                    mAP50_95,
                    mAP75,
                    mAP50,
                    mAP50_95_gt,
                    mAP75_gt,
                    mAP50_gt,
                ) in zip(
                    val_compressed_images_size[seq],
                    val_selections[seq],
                    val_mean_mAP50_95[seq],
                    val_mean_mAP75[seq],
                    val_mean_mAP50[seq],
                    val_mean_gt_mAP50_95[seq],
                    val_mean_gt_mAP75[seq],
                    val_mean_gt_mAP50[seq],
                ):
                    selections = [level for _, level in selections]
                    counts = np.bincount(selections, minlength=5)
                    counts = counts / np.sum(counts) * 100
                    log_entry = (
                        "val",
                        epoch + 1,
                        seq,
                        mAP50_95,
                        mAP75,
                        mAP50,
                        mAP50_95_gt,
                        mAP75_gt,
                        mAP50_gt,
                        compress_img_size,
                        counts,
                        selections,
                    )
                    val_log_writer.writerow(log_entry)
                    vf.flush()

                writer.add_scalar(f"train-loss/{seq}", np.mean(train_loss[seq]), epoch)
                # writer.add_scalar(
                #     f"train-loss1/{seq}", np.mean(train_loss1[seq]), epoch
                # )
                # writer.add_scalar(
                #     f"train-loss2/{seq}", np.mean(train_loss2[seq]), epoch
                # )
                writer.add_scalar(f"val-loss/{seq}", np.mean(val_loss[seq]), epoch)
                # writer.add_scalar(f"val-loss1/{seq}", np.mean(val_loss1[seq]), epoch)
                # writer.add_scalar(f"val-loss2/{seq}", np.mean(val_loss2[seq]), epoch)
                writer.add_scalar(
                    f"train-mAP50_95/{seq}", np.mean(train_mean_mAP50_95[seq]), epoch
                )
                writer.add_scalar(
                    f"train-mAP75/{seq}", np.mean(train_mean_mAP75[seq]), epoch
                )
                writer.add_scalar(
                    f"train-mAP50/{seq}", np.mean(train_mean_mAP50[seq]), epoch
                )
                writer.add_scalar(
                    f"val-mAP50_95/{seq}", np.mean(val_mean_mAP50_95[seq]), epoch
                )
                writer.add_scalar(
                    f"val-mAP75/{seq}", np.mean(val_mean_mAP75[seq]), epoch
                )
                writer.add_scalar(
                    f"val-mAP50/{seq}", np.mean(val_mean_mAP50[seq]), epoch
                )
                writer.add_scalar(
                    f"val-mAP50_95_gt/{seq}", np.mean(val_mean_gt_mAP50_95[seq]), epoch
                )
                writer.add_scalar(
                    f"val-mAP75_gt/{seq}", np.mean(val_mean_gt_mAP75[seq]), epoch
                )
                writer.add_scalar(
                    f"val-mAP50_gt/{seq}", np.mean(val_mean_gt_mAP50[seq]), epoch
                )
                writer.add_text(
                    f"train-selections/{seq}", np.array2string(train_counts), epoch
                )
                writer.add_text(
                    f"val-selections/{seq}", np.array2string(val_counts), epoch
                )
                writer.flush()

                # run.log(
                #     {
                #         f"train-loss/{seq}": np.mean(train_loss[seq]),
                #         # f"train-loss1/{seq}": np.mean(train_loss1[seq]),
                #         # f"train-loss2/{seq}": np.mean(train_loss2[seq]),
                #         f"val-loss/{seq}": np.mean(val_loss[seq]),
                #         # f"val-loss1/{seq}": np.mean(val_loss1[seq]),
                #         # f"val-loss2/{seq}": np.mean(val_loss2[seq]),
                #         f"train-mAP50_95/{seq}": np.mean(train_mean_mAP50_95[seq]),
                #         f"train-mAP75/{seq}": np.mean(train_mean_mAP75[seq]),
                #         f"train-mAP50/{seq}": np.mean(train_mean_mAP50[seq]),
                #         f"val-mAP50_95/{seq}": np.mean(val_mean_mAP50_95[seq]),
                #         f"val-mAP75/{seq}": np.mean(val_mean_mAP75[seq]),
                #         f"val-mAP50/{seq}": np.mean(val_mean_mAP50[seq]),
                #         f"val-mAP50_95_gt/{seq}": np.mean(val_mean_gt_mAP50_95[seq]),
                #         f"val-mAP75_gt/{seq}": np.mean(val_mean_gt_mAP75[seq]),
                #         f"val-mAP50_gt/{seq}": np.mean(val_mean_gt_mAP50[seq]),
                #         f"train-selections/{seq}": train_counts,
                #         f"val-selections/{seq}": val_counts,
                #     }
                # )

            epoch_map50_90 = np.mean(
                [np.mean(val_mean_mAP50_95[seq]) for seq in DATASETS]
            )
            epoch_mean50_90_gt = np.mean(
                [np.mean(val_mean_gt_mAP50_95[seq]) for seq in DATASETS]
            )
            delta = epoch_map50_90 - epoch_mean50_90_gt
            # if delta > max_map50_95:
            #     max_map50_95 = delta
            torch.save(
                model.state_dict(),
                os.path.join(checkpoint, f"{epoch}-{delta}.pth"),
            )
            tf.close()
            vf.close()
        # join the process to ensure the log is written
        dist.barrier()
    if rank == 0:
        writer.close()
    dist.destroy_process_group()
