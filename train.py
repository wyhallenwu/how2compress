import numpy as np
import supervision as sv
import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
from ultralytics import YOLO

import src.utils.metrics as metrics
import wandb
from src.dataset.dataloader import MOTDataset, collate_fn
from src.model.hgmodel import MobileVitV2
from src.utils.cals import macroblocks_wh

DATASETS = [
    "MOT17-02",
    "MOT17-04",
    "MOT17-05",
    "MOT17-09",
    "MOT17-10",
    "MOT17-11",
    "MOT17-13",
]

QP_SET = [30, 34, 37, 41, 45]
EPOCH = 100
LR = 1e-3
MIN_LR = 1e-6
BATCH_SIZE = 2
TRAIN_RATIO = 0.7
ROUND = 100
DEVICE = "cuda:1"


if __name__ == "__main__":
    # writer
    wandb.login()
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="cvpr2025",
        tags=["MOT", "V1"],
        # # track hyperparameters and run metadata
        config={
            "epoch": EPOCH,
            "lr": LR,
            "min_lr": MIN_LR,
            "qp_set": QP_SET,
            "batch_size": BATCH_SIZE,
            "datasets": DATASETS,
            "weight_mAP": 10,
            "reward": [1, 1.1, 1.2, 1.3, 1.4],
        },
    )

    # dataset
    dataset = MOTDataset(
        dataset_dir="/how2compress/data/MOT17Det/train",
        reference_dir="/how2compress/data/detections",
        ssim_label_dir="/how2compress/data/ssim_labels",
    )

    # model
    model = MobileVitV2().to(DEVICE)
    inferencer = YOLO(
        "/how2compress/pretrained/best_MOT_1920.pt", verbose=False
    ).to(DEVICE)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    poly_LR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 32, MIN_LR)

    for epoch in range(EPOCH):
        train_detections = {}
        val_detections = {}
        mean_mAP = []
        max_mean_mAP = 0
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
                drop_last=True,
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                collate_fn=collate_fn,
                shuffle=False,
                pin_memory=True,
                drop_last=True,
            )
            height, width = (
                dataset.curr_seq_property["height"],
                dataset.curr_seq_property["width"],
            )
            # config
            # weight_counts = torch.tensor([1, 0.8, 0.6, 0.4, 0.2]).to("cuda")
            reward_mapping = torch.tensor([1, 1.1, 1.2, 1.3, 1.4]).to(DEVICE)
            weight_mAP = torch.tensor([10]).to(DEVICE)

            # train
            for round in range(ROUND):
                model.train()
                train_detections.setdefault(seq, [[], []])
                for images, labels, indices, ssim_labels in tqdm(
                    train_dataloader, desc=f"EPOCH: {epoch+1}|{round+1}, SEQ: {seq}"
                ):
                    mb_w, mb_h = macroblocks_wh(width, height)
                    images = images.to(DEVICE)
                    ssim_labels = ssim_labels.to(DEVICE)
                    ems_map, ems_map_v, selections = model(images)

                    # inference composed images
                    composed_images = dataset.fetch_compose_images(indices, selections)
                    composed_images = [
                        torch.from_numpy(image) / 255.0 for image in composed_images
                    ]
                    retrieved_images = torch.stack(composed_images).float().to(DEVICE)
                    retrieved_images = rearrange(retrieved_images, "b h w c -> b c h w")

                    preds = inferencer.predict(retrieved_images, verbose=False)
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

                    # record result
                    train_detections[seq][0].extend(preds)
                    train_detections[seq][1].extend(labels)

                    loss, loss1, loss2 = model.loss_fn_det(
                        ems_map,
                        ems_map_v,
                        reward_mapping,
                        weight_mAP,
                        mAP,
                        ssim_labels,
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    wandb.log(
                        {
                            f"loss-train/{seq}": loss.item(),
                            f"loss1-train/{seq}": loss1,
                            f"loss2-train/{seq}": loss2,
                            f"mAP50_95-train/{seq}": mAP.map50_95,
                            f"mAP_75-train/{seq}": mAP.map75,
                            f"mAP_50-train/{seq}": mAP.map50,
                        }
                    )
                overall_mAP_train = sv.MeanAveragePrecision.from_detections(
                    train_detections[seq][0], train_detections[seq][1]
                )
                wandb.log(
                    {
                        f"overall mAP_50_95-train/{seq}": overall_mAP_train.map50_95,
                        f"overall mAP_75-train/{seq}": overall_mAP_train.map75,
                        f"overall mAP_50-train/{seq}": overall_mAP_train.map50,
                    }
                )

                # val
                model.eval()
                val_detections.setdefault(seq, [[], []])
                for images, labels, indices, ssim_labels in tqdm(
                    val_dataloader, desc=f"EPOCH: {epoch+ 1}, VAL SEQ: {seq}"
                ):
                    mb_w, mb_h = macroblocks_wh(width, height)
                    images = images.to(DEVICE)
                    ssim_labels = ssim_labels.to(DEVICE)
                    with torch.no_grad():
                        ems_map, ems_map_v, selections = model(images)

                    # inference composed images
                    composed_images = dataset.fetch_compose_images(indices, selections)
                    composed_images = [
                        torch.from_numpy(image) / 255.0 for image in composed_images
                    ]
                    retrieved_images = torch.stack(composed_images).float().to(DEVICE)
                    retrieved_images = rearrange(retrieved_images, "b h w c -> b c h w")

                    preds = inferencer.predict(retrieved_images, verbose=False)
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

                    # record result
                    val_detections[seq][0].extend(preds)
                    val_detections[seq][1].extend(labels)

                    # loss
                    # weight_counts = torch.tensor([1, 0.8, 0.6, 0.4, 0.2]).to("cuda")
                    # reward_mapping = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1]).to("cuda")
                    # weight_mAP = torch.tensor([10]).to("cuda")

                    loss, loss1, loss2 = model.loss_fn_det(
                        ems_map, ems_map_v, reward_mapping, weight_mAP, mAP, ssim_labels
                    )
                    wandb.log(
                        {
                            f"loss-val/{seq}": loss.item(),
                            f"loss1-val/{seq}": loss1,
                            f"loss2-val/{seq}": loss2,
                            f"mAP50_95-val/{seq}": mAP.map50_95,
                            f"mAP_75-val/{seq}": mAP.map75,
                            f"mAP_50-val/{seq}": mAP.map50,
                        }
                    )

                overall_mAP_val = sv.MeanAveragePrecision.from_detections(
                    val_detections[seq][0], val_detections[seq][1]
                )
                mean_mAP.append(overall_mAP_val.map50_95)
                wandb.log(
                    {
                        f"overall mAP_50_95-val/{seq}": overall_mAP_val.map50_95,
                        f"overall mAP_75-val/{seq}": overall_mAP_val.map75,
                        f"overall mAP_50-val/{seq}": overall_mAP_val.map50,
                        "learning_rate": poly_LR.get_last_lr(),
                    }
                )
            if np.mean(mean_mAP) > max_mean_mAP:
                torch.save(
                    model.state_dict(),
                    "/how2compress/pretrained/best_mbvit2_MOT1704.pt",
                )
                max_mean_mAP = np.mean(mean_mAP)
        poly_LR.step()
