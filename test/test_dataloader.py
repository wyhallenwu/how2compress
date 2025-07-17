import random

import cv2
from src.dataset.dataloader import MOTDataset, collate_fn
from torch.utils.data import DataLoader
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

dataset = MOTDataset(
    dataset_dir="/how2compress/data/MOT17Det/train",
    reference_dir="/how2compress/data/detections",
    ssim_label_dir="/how2compress/data/ssim_labels",
)

for seq in DATASETS:
    dataset.load_sequence(seq)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    height, width = (
        dataset.curr_seq_property["height"],
        dataset.curr_seq_property["width"],
    )
    for images, labels, indices, ssim_labels in dataloader:
        mb_w, mb_h = macroblocks_wh(width, height)
        # generate random choice of qp level in [0, 5) for length mb_w * mb_h
        print(f"images type: {type(images)}, images shape: {images.shape}")
        # images.to("cuda")
        selections = [
            [(mb_idx, 45) for mb_idx in range(mb_w * mb_h)]
            for _ in range(dataloader.batch_size)
        ]
        print(indices)
        print(f"ssim labels: {ssim_labels.shape}")
        composed_images = dataset.fetch_compose_images(indices, selections)
        name = dataset.curr_seq_property["name"]
        # for i, image in enumerate(composed_images):
        #     cv2.imwrite(f"/how2compress/{name}_{indices[i]}.png", image)

        break

    print(dataloader.dataset.curr_seq_property)
