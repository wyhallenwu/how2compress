from src.utils import image_ops, cals, load
import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from torcheval.metrics import PeakSignalNoiseRatio
from einops import rearrange

root = "/how2compress/data/pandasH264"
dataset = "02_OCT_Habour"
high_quality_path = os.path.join(root, dataset, "30")
low_quality_path = os.path.join(root, dataset, "34")

high_quality_files = sorted(os.listdir(high_quality_path))
high_quality_files = [
    os.path.join(high_quality_path, file) for file in high_quality_files
]
low_quality_files = sorted(os.listdir(low_quality_path))
low_quality_files = [os.path.join(low_quality_path, file) for file in low_quality_files]

transform = image_ops.vit_transform_fn()

metric = PeakSignalNoiseRatio()

ref = []
mb_h = 16
mb_w = 16
composed = []
for high_quality_file, low_quality_file in zip(high_quality_files, low_quality_files):
    high_quality_images = load.load_h264_training(high_quality_file)
    high_quality_images = image_ops.wrap_img(high_quality_images)
    cv2.imwrite("high_quality.png", high_quality_images)
    high_quality_images = transform(high_quality_images)
    # print(high_quality_images.shape)
    high_quality_images = rearrange(
        high_quality_images,
        "c (h mb_h) (w mb_w) -> (h w) c mb_h mb_w",
        mb_h=mb_h,
        mb_w=mb_w,
    )
    # ref.append(high_quality_images)
    low_quality_images = load.load_h264_training(low_quality_file)
    low_quality_images = image_ops.wrap_img(low_quality_images)
    cv2.imwrite("low_quality.png", low_quality_images)
    low_quality_images = transform(low_quality_images)
    low_quality_images = rearrange(
        low_quality_images,
        "c (h mb_h) (w mb_w) -> (h w) c mb_h mb_w",
        mb_h=mb_h,
        mb_w=mb_w,
    )
    psnr_diff = []
    for high, low in zip(high_quality_images, low_quality_images):
        # print(low.shape)
        metric.update(low, high)
        ref.append(metric.compute())
        # composed.append(low_quality_images)
    break

# ref = torch.stack(ref).to("cuda:0")
# composed = torch.stack(composed).to("cuda:0")
# ssim_diff, ssim_loss = cals.mb_ssim(composed, ref)
# ssim_diff = ssim_diff.view(2560 // 16, 1440 // 16).cpu().numpy()
# print(f"ssim diff: {ssim_diff.shape}, ssim loss: {ssim_loss}")

print(len(ref))
ref = torch.stack(ref).view(1440 // 16, 2560 // 16)
plt.figure(figsize=(120, 80))
sns.heatmap(ref, cmap="coolwarm", cbar=True, annot=True, fmt=".2f")
plt.tight_layout()
plt.savefig("pnsr_diff.pdf")
