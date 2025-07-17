from src.utils import image_ops, cals, load
import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import LinearSegmentedColormap


DATASET = ["MOT17-02", "MOT17-04", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]

root1 = "/how2compress/data/UNI25CHUNK"
root2 = "/how2compress/video-result"

for dataset in DATASET:
    src = os.path.join(root1, dataset)
    unis = [f for f in os.listdir(src) if f.startswith("uni")]
    unis = sorted([os.path.join(src, f) for f in unis])
    aqs = [f for f in os.listdir(src) if f.startswith("aq")]
    aqs = sorted([os.path.join(src, f) for f in aqs])

    exp_src = os.path.join(root2, dataset)
    accmpegs = [f for f in os.listdir(exp_src) if f.startswith("h264-accmpeg")]
    accmpegs = sorted([os.path.join(exp_src, f) for f in accmpegs])

    ours = [f for f in os.listdir(exp_src) if f.startswith("h264-ours")]
    ours = sorted([os.path.join(exp_src, f) for f in ours])

    unis_tensor = []
    aqs_tensor = []
    accmpegs_tensor = []
    ours_tensor = []

    for uni, aq, accmpeg, our in zip(unis, aqs, accmpegs, ours):
        uni_frames = load.load_mp4_training(uni)
        aq_frames = load.load_mp4_training(aq)
        accmpeg_frames = load.load_mp4_training(accmpeg)
        our_frames = load.load_mp4_training(our)

        uni_frames = image_ops.wrap_img(uni_frames)
        aq_frames = image_ops.wrap_img(aq_frames)
        accmpeg_frames = image_ops.wrap_img(accmpeg_frames)
        our_frames = image_ops.wrap_img(our_frames)

        uni_frames = image_ops.vit_transform_fn()(uni_frames)
        aq_frames = image_ops.vit_transform_fn()(aq_frames)
        accmpeg_frames = image_ops.vit_transform_fn()(accmpeg_frames)
        our_frames = image_ops.vit_transform_fn()(our_frames)

        unis_tensor.append(uni_frames)
        aqs_tensor.append(aq_frames)
        accmpegs_tensor.append(accmpeg_frames)
        ours_tensor.append(our_frames)

    unis_tensor = torch.stack(unis_tensor).to("cuda:0")
    aqs_tensor = torch.stack(aqs_tensor).to("cuda:0")
    accmpegs_tensor = torch.stack(accmpegs_tensor).to("cuda:0")
    ours_tensor = torch.stack(ours_tensor).to("cuda:0")
    assert (
        unis_tensor.shape
        == aqs_tensor.shape
        == accmpegs_tensor.shape
        == ours_tensor.shape
    ), f"unis_tensor.shape={unis_tensor.shape}, aqs_tensor.shape={aqs_tensor.shape}, accmpegs_tensor.shape={accmpegs_tensor.shape}, ours_tensor.shape={ours_tensor.shape}"

    ssim_diff_uni_aq, ssim_loss_uni_aq = cals.mb_ssim(unis_tensor, aqs_tensor)
    ssim_diff_uni_accmpeg, ssim_loss_uni_accmpeg = cals.mb_ssim(
        unis_tensor, accmpegs_tensor
    )
    ssim_diff_uni_ours, ssim_loss_uni_ours = cals.mb_ssim(unis_tensor, ours_tensor)

    print(
        f"{dataset}-uni_aq: 95% {torch.quantile(ssim_diff_uni_aq.view(-1), 0.95)}, 90% {torch.quantile(ssim_diff_uni_aq.view(-1), 0.9)}, 85% {torch.quantile(ssim_diff_uni_aq.view(-1), 0.85)}, 80% {torch.quantile(ssim_diff_uni_aq.view(-1), 0.8)}, 75% {torch.quantile(ssim_diff_uni_aq.view(-1), 0.75)}, 70% {torch.quantile(ssim_diff_uni_aq.view(-1), 0.7)}, 65% {torch.quantile(ssim_diff_uni_aq.view(-1), 0.65)}, 60% {torch.quantile(ssim_diff_uni_aq.view(-1), 0.6)}, 55% {torch.quantile(ssim_diff_uni_aq.view(-1), 0.55)}, 50% {torch.quantile(ssim_diff_uni_aq.view(-1), 0.5)}"
    )
    # print(
    #     f"{dataset}-uni_accmpeg: 95% {torch.quantile(ssim_diff_uni_accmpeg.view(-1), 0.95)}, 90% {torch.quantile(ssim_diff_uni_accmpeg.view(-1), 0.9)}, 85% {torch.quantile(ssim_diff_uni_accmpeg.view(-1), 0.85)}, 80% {torch.quantile(ssim_diff_uni_accmpeg.view(-1), 0.8)}, 75% {torch.quantile(ssim_diff_uni_accmpeg.view(-1), 0.75)}, 70% {torch.quantile(ssim_diff_uni_accmpeg.view(-1), 0.7)}, 65% {torch.quantile(ssim_diff_uni_accmpeg.view(-1), 0.65)}, 60% {torch.quantile(ssim_diff_uni_accmpeg.view(-1), 0.6)}, 55% {torch.quantile(ssim_diff_uni_accmpeg.view(-1), 0.55)}, 50% {torch.quantile(ssim_diff_uni_accmpeg.view(-1), 0.5)}"
    # )
    # print(
    #     f"{dataset}-uni_ours: 95% {torch.quantile(ssim_diff_uni_ours.view(-1), 0.95)}, 90% {torch.quantile(ssim_diff_uni_ours.view(-1), 0.9)}, 85% {torch.quantile(ssim_diff_uni_ours.view(-1), 0.85)}, 80% {torch.quantile(ssim_diff_uni_ours.view(-1), 0.8)}, 75% {torch.quantile(ssim_diff_uni_ours.view(-1), 0.75)}, 70% {torch.quantile(ssim_diff_uni_ours.view(-1), 0.7)}, 65% {torch.quantile(ssim_diff_uni_ours.view(-1), 0.65)}, 60% {torch.quantile(ssim_diff_uni_ours.view(-1), 0.6)}, 55% {torch.quantile(ssim_diff_uni_ours.view(-1), 0.55)}, 50% {torch.quantile(ssim_diff_uni_ours.view(-1), 0.5)}"
    # )

    print(f"{dataset}-uni_aq: {ssim_diff_uni_aq.mean()}")
    print(f"{dataset}-uni_accmpeg: {ssim_diff_uni_accmpeg.mean()}")
    print(f"{dataset}-uni_ours: {ssim_diff_uni_ours.mean()}")

    percentiles = [0.25, 0.50, 0.75, 1.00]

    uni_aq_percentiles = torch.quantile(ssim_diff_uni_aq.view(-1), torch.tensor(percentiles, device=ssim_diff_uni_aq.device))
    uni_accmpeg_percentiles = torch.quantile(ssim_diff_uni_accmpeg.view(-1), torch.tensor(percentiles, device=ssim_diff_uni_accmpeg.device))
    uni_ours_percentiles = torch.quantile(ssim_diff_uni_ours.view(-1), torch.tensor(percentiles, device=ssim_diff_uni_ours.device))

    print(f"{dataset}-uni_aq percentiles (25%, 50%, 75%, 100%): {uni_aq_percentiles.tolist()}")
    print(f"{dataset}-uni_accmpeg percentiles (25%, 50%, 75%, 100%): {uni_accmpeg_percentiles.tolist()}")
    print(f"{dataset}-uni_ours percentiles (25%, 50%, 75%, 100%): {uni_ours_percentiles.tolist()}")

    # Compute (max - min) / 4 * [1, 2, 3, 4]
    def compute_ssim_range_percentages(ssim_diff_tensor, label):
        ssim_flat = ssim_diff_tensor.view(-1)
        ssim_min = torch.min(ssim_flat)
        ssim_max = torch.max(ssim_flat)
        interval = (ssim_max - ssim_min) / 10

        thresholds = [ssim_min + interval * i for i in range(1, 10)]

        percentages = [(ssim_flat < t).float().mean().item() * 100 for t in thresholds]

        print(f"{label}:")
        for i, (t, p) in enumerate(zip(thresholds, percentages), 1):
            print(f"  Threshold {i} = {t.item():.6f}, Percentage below = {p:.2f}%")

    compute_ssim_range_percentages(ssim_diff_uni_aq, f"{dataset}-uni_aq")
    compute_ssim_range_percentages(ssim_diff_uni_accmpeg, f"{dataset}-uni_accmpeg")
    compute_ssim_range_percentages(ssim_diff_uni_ours, f"{dataset}-uni_ours")
