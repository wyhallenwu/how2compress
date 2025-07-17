import os
import shutil

from tqdm import tqdm
import numpy as np
import torch
from einops import rearrange
from src.utils import cals, image_ops, load
import cv2

ROOT = "/how2compress/data/detections"
DEST = "/how2compress/data/ssim_labels"
QP_SET = [45, 41, 37, 34, 30]
# QP_MAP = {30: 4, 34: 3, 37: 2, 41: 1, 45: 0}

if not os.path.exists(DEST):
    os.makedirs(DEST)

for seq in sorted(os.listdir(ROOT)):
    # seq = "MOT17-11"
    length = len(os.listdir(os.path.join(ROOT, seq, str(QP_SET[0]))))
    # print(f"Processing {seq} with {length} frames")
    for i in tqdm(range(length), desc=seq):
        frames = []
        ref_frames = []
        for qp in QP_SET:
            src = os.path.join(ROOT, seq, str(qp), f"{i+1:06d}.h264")
            frame = load.load_h264_training(src)
            frame = image_ops.wrap_img(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
            frames.append(frame)
            if qp == 30:
                ref_frames.extend([frame for _ in range(len(QP_SET))])
        frames = torch.from_numpy(np.stack(frames, axis=0)).float().to("cuda")
        # frames = rearrange(frames, "B H W C -> B C H W")
        ref_frames = torch.from_numpy(np.stack(ref_frames, axis=0)).float().to("cuda")
        # ref_frames = rearrange(ref_frames, "B H W C -> B C H W")
        assert frames.shape == ref_frames.shape
        # print(f"frames shape: {frames.shape}, ref_frame shape: {ref_frames.shape}")
        ssim_val, ssim_loss = cals.mb_ssim(ref_frames, frames)
        ssim_label = cals.get_indices(ssim_val)
        # print(f"ssim_val: {ssim_val.shape}, ssim_loss: {ssim_loss}")
        # print(f"ssim label: {ssim_label.shape}, shape: {ssim_label.shape}")
        counts = torch.bincount(ssim_label, minlength=5)
        # print(f"counts: {counts}")
        with open(os.path.join(DEST, f"{seq}.csv"), "a") as f:
            np.savetxt(
                f, ssim_label.cpu().numpy().reshape(1, -1), delimiter=",", fmt="%d"
            )
