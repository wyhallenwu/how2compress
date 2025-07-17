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
from src.utils.cals import macroblocks_wh
from src.utils.cals import mb_ssim
import time
import os
import csv

DATASETS = [
    "MOT17-02",
    "MOT17-04",
    "MOT17-05",
    "MOT17-09",
    "MOT17-10",
    "MOT17-11",
    "MOT17-13",
]
QP_MAPPING = {45: 0, 41: 1, 37: 2, 34: 3, 30: 4}

QP_SET = [30, 34, 37, 41, 45]
EPOCH = 100
LR = 1e-2
MIN_LR = 1e-3
BATCH_SIZE = 16
TRAIN_RATIO = 0.7
DEVICE = "cuda:0"
REWARD_MAPPING = [1, 1.5, 2, 2.5, 3]
WEIGHT_MAP = [10]
ACCUMULATE_GRAD = 8
RESIZE_FACTOR = 4


def train_epoch(
    model: MobileVitV2,
    inferencer: YOLO,
    optimizer: torch.optim.Optimizer,
    dataset: MOTDataset,
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

    train_loss = {}
    train_loss1 = {}
    train_loss2 = {}

    val_loss = {}
    val_loss1 = {}
    val_loss2 = {}

    train_compressed_images_size = {}
    val_compressed_images_size = {}

    for seq in DATASETS:
        # split into train and val
        dataset.load_sequence(seq)
        train_size = int(TRAIN_RATIO * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, test_size])

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=collate_fn,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        height, width = (
            dataset.curr_seq_property["height"],
            dataset.curr_seq_property["width"],
        )
        # config
        reward_mapping = torch.exp(torch.tensor(REWARD_MAPPING)).to(DEVICE)
        weight_mAP = torch.tensor(WEIGHT_MAP).to(DEVICE)

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
        train_detections.setdefault(seq, [[], []])
        val_detections.setdefault(seq, [[], []])
        train_selections.setdefault(seq, [])
        val_selections.setdefault(seq, [])
        train_compressed_images_size.setdefault(seq, [])
        val_compressed_images_size.setdefault(seq, [])

        model.train()
        step = 1
        mb_w, mb_h = macroblocks_wh(width, height)
        resizer = resize_img_tensor((mb_h * 4, mb_w * 4))
        model.set_output_size((mb_h, mb_w))

        for images, labels, indices in tqdm(
            train_dataloader, desc=f"[TRAIN] EPOCH: {epoch+1}, SEQ: {seq}"
        ):
            images = images.to(DEVICE)
            ems_map, ems_map_v, selections = model(resizer(images))

            # inference composed images
            compressed_images, compressed_images_size = dataset.enc_and_ret(
                indices, selections
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
                    ),
                )
                for det in preds
            ]
            # FIXME: should we use the detections from the original images instead of gt?
            mAP = sv.MeanAveragePrecision.from_detections(preds, labels)

            ssim_diffs, _ = mb_ssim(compressed_images, images)
            adjust_indices = model.get_adjusted_labels(
                labels, ems_map, ssim_diffs, mb_h, mb_w
            )
            selections = ems2selections(adjust_indices.cpu().numpy())

            loss, loss1, loss2 = model.loss_fn_det(
                ems_map,
                ems_map_v,
                reward_mapping,
                weight_mAP,
                mAP,
                adjust_indices,
            )
            # loss = loss / ACCUMULATE_GRAD
            # loss.backward()

            # if step % ACCUMULATE_GRAD == 0:
            #     optimizer.step()
            #     optimizer.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            # log
            train_loss[seq].append(loss.item())
            train_loss1[seq].append(loss1)
            train_loss2[seq].append(loss2)
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
            val_dataloader, desc=f"[VAL] EPOCH: {epoch + 1}, SEQ: {seq}"
        ):
            images = images.to(DEVICE)
            # ssim_labels = ssim_labels.to(DEVICE)
            with torch.no_grad():
                ems_map, ems_map_v, selections = model(resizer(images))

            # inference composed images

            compressed_images, compressed_images_size = dataset.enc_and_ret(
                indices, selections
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
                    ),
                )
                for det in preds
            ]
            mAP = sv.MeanAveragePrecision.from_detections(preds, labels)

            ssim_diffs, _ = mb_ssim(compressed_images, images)
            adjust_indices = model.get_adjusted_labels(
                labels, ems_map, ssim_diffs, mb_h, mb_w
            )
            selections = ems2selections(adjust_indices.cpu().numpy())

            loss, loss1, loss2 = model.loss_fn_det(
                ems_map,
                ems_map_v,
                reward_mapping,
                weight_mAP,
                mAP,
                adjust_indices,
            )
            # log
            val_loss[seq].append(loss.item())
            val_loss1[seq].append(loss1)
            val_loss2[seq].append(loss2)
            val_mean_mAP50_95[seq].append(mAP.map50_95)
            val_mean_mAP75[seq].append(mAP.map75)
            val_mean_mAP50[seq].append(mAP.map50)
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
        train_detections,
        val_detections,
        train_selections,
        val_selections,
        train_compressed_images_size,
        val_compressed_images_size,
    )


if __name__ == "__main__":
    # writer
    wandb.login()
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="cvpr2025",
        tags=["MOT", "V1"],
        group="DDP1",
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
            "threshold inside": 0.7,
            "threshold outside": 0.7,
        },
    )

    # dataset
    dataset = MOTDataset(
        dataset_dir="/how2compress/data/MOT17Det/train",
        reference_dir="/how2compress/data/detections",
        ssim_label_dir="/how2compress/data/ssim_labels",
        yuv_dir="/how2compress/data/MOT17DetYUV",
        resize_factor=RESIZE_FACTOR,
    )

    # model
    model = MobileVitV2().to(DEVICE)
    inferencer = YOLO(
        "/how2compress/pretrained/best_MOT_1920.pt", verbose=False
    ).to(DEVICE)

    # checkpoint
    folder = "/how2compress/pretrained/train"
    start_time = time.time()
    # checkpoint = os.path.join(folder, str(start_time))
    curr_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    checkpoint = os.path.join(folder, curr_time)
    if os.path.exists(folder):
        os.makedirs(checkpoint)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    poly_LR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 32, MIN_LR)
    min_loss = np.inf

    vf = open(os.path.join(checkpoint, "val_log.txt"), "a")
    tf = open(os.path.join(checkpoint, "train_log.txt"), "a")

    train_log_writer = csv.writer(tf)
    val_log_writer = csv.writer(vf)

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
            train_detections,
            val_detections,
            train_selections,
            val_selections,
            train_compressed_images_size,
            val_compressed_images_size,
        ) = train_epoch(model, inferencer, optimizer, dataset)

        poly_LR.step()
        for seq in DATASETS:
            train_counts = []
            for selections in train_selections[seq]:
                for _, level in selections:
                    train_counts.append(QP_MAPPING[level])
            train_counts = np.bincount(train_counts, minlength=5)
            train_counts = train_counts / np.sum(train_counts) * 100
            val_counts = []
            for selections in val_selections[seq]:
                for _, level in selections:
                    val_counts.append(QP_MAPPING[level])
            val_counts = np.bincount(val_counts, minlength=5)
            val_counts = val_counts / np.sum(val_counts) * 100

            for compress_img_size, selections, mAP50_95, mAP75, mAP50 in zip(
                train_compressed_images_size[seq],
                train_selections[seq],
                train_mean_mAP50_95[seq],
                train_mean_mAP75[seq],
                train_mean_mAP50[seq],
            ):
                selections = [QP_MAPPING[level] for _, level in selections]
                counts = np.bincount(selections, minlength=5)
                counts = counts / np.sum(counts) * 100
                log_entry = (
                    epoch + 1,
                    seq,
                    "train",
                    mAP50_95,
                    mAP75,
                    mAP50,
                    compress_img_size,
                    counts,
                    selections,
                )
                train_log_writer.writerow(log_entry)
                tf.flush()
            for compress_img_size, selections, mAP50_95, mAP75, mAP50 in zip(
                val_compressed_images_size[seq],
                val_selections[seq],
                val_mean_mAP50_95[seq],
                val_mean_mAP75[seq],
                val_mean_mAP50[seq],
            ):
                selections = [QP_MAPPING[level] for _, level in selections]
                counts = np.bincount(selections, minlength=5)
                counts = counts / np.sum(counts) * 100
                log_entry = (
                    epoch + 1,
                    seq,
                    "val",
                    mAP50_95,
                    mAP75,
                    mAP50,
                    compress_img_size,
                    counts,
                    selections,
                )
                val_log_writer.writerow(log_entry)
                vf.flush()

            run.log(
                {
                    f"train-loss/{seq}": np.mean(train_loss[seq]),
                    f"train-loss1/{seq}": np.mean(train_loss1[seq]),
                    f"train-loss2/{seq}": np.mean(train_loss2[seq]),
                    f"val-loss/{seq}": np.mean(val_loss[seq]),
                    f"val-loss1/{seq}": np.mean(val_loss1[seq]),
                    f"val-loss2/{seq}": np.mean(val_loss2[seq]),
                    f"train-mAP50_95/{seq}": np.mean(train_mean_mAP50_95[seq]),
                    f"train-mAP75/{seq}": np.mean(train_mean_mAP75[seq]),
                    f"train-mAP50/{seq}": np.mean(train_mean_mAP50[seq]),
                    f"val-mAP50_95/{seq}": np.mean(val_mean_mAP50_95[seq]),
                    f"val-mAP75/{seq}": np.mean(val_mean_mAP75[seq]),
                    f"val-mAP50/{seq}": np.mean(val_mean_mAP50[seq]),
                    f"train-selections/{seq}": train_counts,
                    f"val-selections/{seq}": val_counts,
                }
            )

        epoch_mean_loss = np.mean([np.mean(val_loss[seq]) for seq in DATASETS])
        if epoch_mean_loss < min_loss:
            min_loss = epoch_mean_loss
            torch.save(
                model.state_dict(),
                os.path.join(checkpoint, f"{epoch}-best.pth"),
            )
    tf.close()
    vf.close()
