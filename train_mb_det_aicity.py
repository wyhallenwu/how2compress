import numpy as np
import supervision as sv
import torch
from src.utils.image_ops import resize_img_tensor
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from ultralytics import YOLO

import src.utils.metrics as metrics
from src.model.utils import ems2selections
import wandb
from src.dataset.dataloader import MOTDataset, collate_fn, AicityDataset
from src.model.hgmodel import MobileVitV2
from src.utils.cals import macroblocks_wh
from src.utils.cals import mb_ssim
import yaml
from easydict import EasyDict as edict
import argparse
import os
import csv
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


QMIN = 30
QMAX = 45

VIDEO_CLIPS1 = [
    ("S01", "c001"),
    ("S01", "c002"),
    ("S01", "c003"),
    ("S01", "c004"),
    ("S01", "c005"),
]

VIDEO_CLIPS2 = [
    ("S03", "c010"),
    ("S03", "c011"),
    ("S03", "c012"),
    ("S03", "c013"),
    ("S03", "c014"),
    ("S03", "c015"),
]

VIDEO_CLIPS3 = [
    ("S04", "c016"),
    ("S04", "c017"),
    ("S04", "c018"),
    ("S04", "c019"),
    ("S04", "c020"),
    ("S04", "c021"),
    ("S04", "c022"),
    ("S04", "c023"),
    ("S04", "c024"),
    ("S04", "c025"),
    ("S04", "c026"),
    ("S04", "c027"),
    ("S04", "c028"),
    ("S04", "c029"),
    # ("S04", "c030"),
    # ("S04", "c031"),
    # ("S04", "c032"),
    ("S04", "c033"),
    ("S04", "c034"),
    ("S04", "c035"),
    ("S04", "c036"),
    ("S04", "c037"),
    ("S04", "c038"),
    ("S04", "c039"),
    # ("S04", "c040"),
]


def train_epoch(
    model: MobileVitV2,
    inferencer: YOLO,
    optimizer: torch.optim.Optimizer,
    dataset: AicityDataset,
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
        # split into train and val
        dataset.load_sequence(seq)
        train_size = int(TRAIN_RATIO * len(dataset))
        # test_size = len(dataset) - train_size
        train_indices = list(range(0, train_size))
        val_indices = list(range(train_size, len(dataset)))
        # train_dataset, val_dataset = random_split(dataset, [train_size, test_size])
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

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
        reward_mapping = torch.exp(torch.tensor(REWARD_MAPPING)).to(device_id)
        weight_mAP = torch.tensor(WEIGHT_MAP).to(device_id)

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
        model.set_output_size((mb_h, mb_w))

        for images, labels, indices in tqdm(
            train_dataloader, desc=f"[TRAIN] EPOCH: {curr_epoch+1}, SEQ: {seq}"
        ):
            images = images.to(device_id)
            ems_map, ems_map_v, selections = model(resizer(images))

            targets = inferencer.predict(images, classes=[2], conf=0.6, verbose=False)
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

            compressed_images, _ = dataset.enc_and_ret(
                indices, selections, device_id, QMIN, QMAX
            )
            compressed_images = compressed_images.to(device_id)

            ssim_diffs, _ = mb_ssim(compressed_images, images)
            adjust_indices = model.get_adjusted_labels(
                labels,
                ems_map,
                ssim_diffs,
                mb_h,
                mb_w,
                max(EXPLORE_PROB - EXPLORE_DECAY * step, 0.1),
            )
            selections = ems2selections(adjust_indices.cpu().numpy())
            compressed_images, compressed_images_size = dataset.enc_and_ret(
                indices, selections, device_id, QMIN, QMAX
            )
            compressed_images = compressed_images.to(device_id)

            preds = inferencer.predict(
                compressed_images, classes=[2], conf=0.5, verbose=False
            )
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
            mAP = sv.MeanAveragePrecision.from_detections(preds, targets)

            loss, loss1, loss2 = model.loss_fn_det(
                ems_map,
                ems_map_v,
                reward_mapping,
                weight_mAP,
                mAP,
                adjust_indices,
            )

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
            val_dataloader, desc=f"[VAL] EPOCH: {curr_epoch + 1}, SEQ: {seq}"
        ):
            images = images.to(device_id)
            with torch.no_grad():
                ems_map, ems_map_v, selections = model(resizer(images))

            targets = inferencer.predict(images, classes=[2], verbose=False)
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

            # when val the model, use direct predict selections
            compressed_images, _ = dataset.enc_and_ret(
                indices, selections, device_id, QMIN, QMAX
            )
            compressed_images = compressed_images.to(device_id)

            ssim_diffs, _ = mb_ssim(compressed_images, images)
            adjust_indices = model.get_adjusted_labels(
                labels, ems_map, ssim_diffs, mb_h, mb_w
            )
            # selections = ems2selections(adjust_indices.cpu().numpy())
            # compressed_images, compressed_images_size = dataset.enc_and_ret(
            #     indices, selections, device_id
            # )

            preds = inferencer.predict(compressed_images, classes=[2], verbose=False)
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
            mAP = sv.MeanAveragePrecision.from_detections(preds, targets)
            # mAP_gt = sv.MeanAveragePrecision.from_detections(targets, labels)

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
            # val_mean_gt_mAP50_95[seq].append(mAP_gt.map50_95)
            # val_mean_gt_mAP75[seq].append(mAP_gt.map75)
            # val_mean_gt_mAP50[seq].append(mAP_gt.map50)
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


# def setup_log():
#     # writer
#     wandb.login()
#     run = wandb.init(
#         # set the wandb project where this run will be logged
#         project="cvpr2025",
#         tags=["MOT", "V1"],
#         group="DDP1",
#         # # track hyperparameters and run metadata
#         config={
#             "epoch": EPOCH,
#             "lr": LR,
#             "min_lr": MIN_LR,
#             "qp_set": QP_SET,
#             "batch_size": BATCH_SIZE,
#             "resize_factor": RESIZE_FACTOR,
#             "datasets": DATASETS,
#             "weight mAP": WEIGHT_MAP,
#             "reward mapping": REWARD_MAPPING,
#             "threshold inside": THRESHOLD_INSIDE,
#             "threshold outside": THRESHOLD_OUTSIDE,
#         },
#     )
#     return run


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, help="training config file")
    # args = parser.parse_args()
    # with open(args.config, "r") as f:
    #     config = yaml.safe_load(f)
    # config = edict(config)

    # NAME = config.NAME
    # DATASETS = config.DATASETS
    # EPOCH = config.EPOCH
    # LR = config.LR
    # MIN_LR = config.MIN_LR
    # BATCH_SIZE = config.BATCH_SIZE
    # TRAIN_RATIO = config.TRAIN_RATIO
    # REWARD_MAPPING = config.REWARD_MAPPING
    # WEIGHT_MAP = config.WEIGHT_MAP
    # RESIZE_FACTOR = config.RESIZE_FACTOR
    # THRESHOLD_INSIDE = config.THRESHOLD_INSIDE
    # THRESHOLD_OUTSIDE = config.THRESHOLD_OUTSIDE
    # EXPLORE_PROB = config.EXPLORE_PROB
    # EXPLORE_DECAY = config.EXPLORE_DECAY

    NAME = "exp-aicity-S03"
    DATASETS = VIDEO_CLIPS3
    EPOCH = 3
    LR = 1e-3
    MIN_LR = 1e-6
    BATCH_SIZE = 4
    TRAIN_RATIO = 0.7
    REWARD_MAPPING = [1, 1.3, 1.6, 1.9, 2.2]
    WEIGHT_MAP = [10]
    RESIZE_FACTOR = 4
    THRESHOLD_INSIDE = 0.99  # tbd
    THRESHOLD_OUTSIDE = 0.96  # tbd
    EXPLORE_PROB = 0.8
    EXPLORE_DECAY = 0.1

    # init distributed
    # run = setup_log()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_id = rank % torch.cuda.device_count()
    # model
    model = MobileVitV2(
        threshold_inside=THRESHOLD_INSIDE, threshold_outside=THRESHOLD_OUTSIDE
    ).to(device_id)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    inferencer = YOLO("yolov8m.pt", verbose=False).to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=LR)

    poly_LR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 32, MIN_LR)
    max_map50_95 = -np.inf

    # checkpoint
    folder = "/home/yuheng/tetris/pretrained/train"
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

    # dataset = MOTDataset(
    #     dataset_dir="/home/yuheng/cvpr2025/data/MOT17Det/train",
    #     reference_dir="/home/yuheng/cvpr2025/data/detections",
    #     # ssim_label_dir="/home/yuheng/cvpr2025/data/ssim_labels",
    #     yuv_dir="/home/yuheng/cvpr2025/data/MOT17DetYUV",
    #     resize_factor=RESIZE_FACTOR,
    # )

    dataset = AicityDataset(
        "/home/yuheng/tetris/data/aicity",
        "/home/yuheng/tetris/data/aicity/H264",
        "/home/yuheng/tetris/data/aicity/YUV",
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
            vf = open(
                os.path.join(
                    checkpoint, f"val_log-{THRESHOLD_INSIDE}-{THRESHOLD_OUTSIDE}.txt"
                ),
                "a",
            )
            tf = open(
                os.path.join(
                    checkpoint, f"train_log-{THRESHOLD_INSIDE}-{THRESHOLD_OUTSIDE}.txt"
                ),
                "a",
            )
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
                        counts.tolist(),
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
                        mAP50_95 - mAP50_95_gt,
                        counts.tolist(),
                        selections,
                    )
                    val_log_writer.writerow(log_entry)
                    vf.flush()
                writer.add_scalar(f"train-loss/{seq}", np.mean(train_loss[seq]), epoch)
                writer.add_scalar(
                    f"train-loss1/{seq}", np.mean(train_loss1[seq]), epoch
                )
                writer.add_scalar(
                    f"train-loss2/{seq}", np.mean(train_loss2[seq]), epoch
                )
                writer.add_scalar(f"val-loss/{seq}", np.mean(val_loss[seq]), epoch)
                writer.add_scalar(f"val-loss1/{seq}", np.mean(val_loss1[seq]), epoch)
                writer.add_scalar(f"val-loss2/{seq}", np.mean(val_loss2[seq]), epoch)
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
                # writer.add_scalar(
                #     f"val-mAP50_95_gt/{seq}", np.mean(val_mean_gt_mAP50_95[seq]), epoch
                # )
                # writer.add_scalar(
                #     f"val-mAP75_gt/{seq}", np.mean(val_mean_gt_mAP75[seq]), epoch
                # )
                # writer.add_scalar(
                #     f"val-mAP50_gt/{seq}", np.mean(val_mean_gt_mAP50[seq]), epoch
                # )
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
                #         f"train-loss1/{seq}": np.mean(train_loss1[seq]),
                #         f"train-loss2/{seq}": np.mean(train_loss2[seq]),
                #         f"val-loss/{seq}": np.mean(val_loss[seq]),
                #         f"val-loss1/{seq}": np.mean(val_loss1[seq]),
                #         f"val-loss2/{seq}": np.mean(val_loss2[seq]),
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
            # epoch_mean50_90_gt = np.mean(
            #     [np.mean(val_mean_mAP50_95[seq]) for seq in DATASETS]
            # )
            # delta = epoch_map50_90
            # if delta > max_map50_95:
            #     max_map50_95 = delta
            torch.save(
                model.state_dict(),
                os.path.join(
                    checkpoint,
                    f"{epoch}-{epoch_map50_90}-{THRESHOLD_INSIDE}-{THRESHOLD_OUTSIDE}.pth",
                ),
            )
            # wandb.save(f"{epoch}-best.pth")
            tf.close()
            vf.close()
        # join the process to ensure the log is written
        dist.barrier()
    if rank == 0:
        writer.close()
    dist.destroy_process_group()
