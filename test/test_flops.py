import torch
import cv2
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model.hgmodel import MobileVitV2
from src.model.utils import ems2selections
from src.utils import image_ops, load, cals, metrics
from src.model.am_model import AccMpeg
from src.dataset.dataloader import MOTDataset, collate_fn
import os
from ultralytics import YOLO
from tqdm import tqdm
import supervision as sv
import time
import argparse
from calflops import calculate_flops
import numpy as np

# parser = argparse.ArgumentParser()
# parser.add_argument("--dataset", type=str, help="dataset name")
# parser.add_argument("--model", type=str, help="model path")
# # parser.add_argument("--results", type=str, help="results path")

# args = parser.parse_args()

DEVICE = "cuda:0"
RESIZE_FACTOR = 4
# DATASET = [args.dataset]
BATCH_SIZE = 1


# model = MobileVitV2()
# model.load_state_dict(
#     torch.load(
#         # "/how2compress/pretrained/train/exp-ddp-30-45-mot17-04/0-best.pth"
#         # "/how2compress/pretrained/train/exp-ddp-30-45-all/0-best.pth"
#         # "/how2compress/pretrained/train/exp-ddp-30-45-1709/0-best.pth"
#         "/how2compress/pretrained/train/exp1710-1/1-0.4620776571761211+-0.0023735605196651965-0.993-0.971.pth"
#         # args.model
#     )
# )
# model.to(DEVICE)
width = 1920
height = 1080
mb_h = height // 16
mb_w = width // 16
# model.set_output_size((mb_h, mb_w))
# # resizer = image_ops.resize_img_tensor((1088 // 16 * 4, 1920 // 16 * 4))


model = AccMpeg(height // 16, width // 16)
# model.load_state_dict(
#     torch.load(
#         "/how2compress/pretrained/train/exp-accmpeg-1711-0.6q/1--0.04199127906485545.pth"
#     )
# )
model.to(DEVICE)


input_shape = (1, 3, mb_h * RESIZE_FACTOR, mb_w * RESIZE_FACTOR)
flops, macs, params = calculate_flops(
    model, input_shape, output_as_string=True, output_precision=4
)
print("FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))


input_tensor = torch.randn(1, 3, mb_h * RESIZE_FACTOR, mb_w * RESIZE_FACTOR).to(DEVICE)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
with torch.no_grad():
    model(input_tensor)
memory_allocated = torch.cuda.max_memory_allocated() / (1024**2)  # in MB

print(f"Peak GPU memory usage: {memory_allocated:.2f} MB")

torch.cuda.empty_cache()
t = []
for i in range(100):
    with torch.no_grad():
        start_t = time.time()
        model(input_tensor)
        end_t = time.time()
    # torch.cuda.empty_cache()
    if i > 20:
        t.append(end_t - start_t)
    # print(end_t - start_t)
print(f"average time: {np.mean(t)}")


# inferencer = YOLO(
#     "/how2compress/pretrained/best_MOT_1920.pt", verbose=False
# ).to(DEVICE)

# transform = image_ops.vit_transform_fn()
# root = "/how2compress/data/MOT17DetH264"
# dataset = args.dataset
# results_root = "/how2compress/results"
# r = os.path.join(results_root, f"eval30-45-{dataset}-f7-1.txt")
# rd = os.path.join(results_root, f"decisions30-45-{dataset}-f7-1.txt")
# path = os.path.join(root, dataset, "30")
# frames = sorted(os.listdir(path))
# frames = [os.path.join(path, frame) for frame in frames]
# enc_frames_dir = os.path.join(results_root, dataset)
# if not os.path.exists(enc_frames_dir):
#     os.makedirs(enc_frames_dir)

# # load gt
# dataset = MOTDataset(
#     dataset_dir="/how2compress/data/MOT17Det/train",
#     reference_dir="/how2compress/data/detections",
#     ssim_label_dir="/how2compress/data/ssim_labels",
#     yuv_dir="/how2compress/data/MOT17DetYUV",
#     resize_factor=RESIZE_FACTOR,
# )

# model.eval()

# mAPs = []
# mAPs_gt = []
# frames_size = []
# times = []
# ret_frames = []
# decisions = []
# for seq in DATASET:
#     dataset.load_sequence(seq)
#     dataloader = DataLoader(
#         dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False
#     )
#     height, width = (
#         dataset.curr_seq_property["height"],
#         dataset.curr_seq_property["width"],
#     )
#     mb_w, mb_h = cals.macroblocks_wh(width, height)
#     resizer = image_ops.resize_img_tensor((mb_h * 4, mb_w * 4))
#     model.set_output_size((mb_h, mb_w))

#     for images, labels, indices in tqdm(dataloader, desc=f"{seq} val"):
#         images = images.to(DEVICE)
#         resize_images = resizer(images)
#         start_time = time.time()
#         ems_map_indices, ems_map_v, selections = model(resize_images)
#         end_time = time.time()
#         times.append(end_time - start_time)
#         targets = inferencer.predict(images, classes=[1], verbose=False)
#         targets = metrics.yolo2sv(targets)
#         targets = [
#             metrics.normalize_detections(
#                 det,
#                 (
#                     dataset.curr_seq_property["width"],
#                     dataset.curr_seq_property["height"],
#                     # images.shape[3],
#                     # images.shape[2],
#                 ),
#             )
#             for det in targets
#         ]
#         ret_selections = [[level for _, level in selection] for selection in selections]
#         decisions.extend(ret_selections)
#         compressed_images, sizes, enc_frames = dataset.enc_and_ret_val(
#             indices, selections, DEVICE
#         )
#         ret_frames.extend(enc_frames)
#         compressed_images = compressed_images.to(DEVICE)
#         preds = inferencer.predict(compressed_images, classes=[1], verbose=False)
#         preds = metrics.yolo2sv(preds)
#         preds = [
#             metrics.normalize_detections(
#                 det,
#                 (
#                     dataset.curr_seq_property["width"],
#                     dataset.curr_seq_property["height"],
#                     # images.shape[3],
#                     # images.shape[2],
#                 ),
#             )
#             for det in preds
#         ]
#         assert len(preds) == len(
#             targets
#         ), f"preds size {len(preds)} != targets size {len(targets)}"

#         frames_size.extend(sizes)
#         mAP_t = sv.MeanAveragePrecision.from_detections(targets, labels)
#         mAP = sv.MeanAveragePrecision.from_detections(preds, labels)
#         mAPs.append(mAP)
#         mAPs_gt.append(mAP_t)
