from src.utils import image_ops, cals, load
import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import LinearSegmentedColormap

root = "/how2compress/data/MOT17DetH264"
dataset = "MOT17-04"
low_qp = "45"
high_quality_path = os.path.join(root, dataset, "30")
low_quality_path = os.path.join(root, dataset, low_qp)

high_quality_files = sorted(os.listdir(high_quality_path))
high_quality_files = [
    os.path.join(high_quality_path, file) for file in high_quality_files
]
low_quality_files = sorted(os.listdir(low_quality_path))
low_quality_files = [os.path.join(low_quality_path, file) for file in low_quality_files]

transform = image_ops.vit_transform_fn()

ref = []
composed = []
for high_quality_file, low_quality_file in zip(high_quality_files, low_quality_files):
    high_quality_images = load.load_h264_training(high_quality_file)
    high_quality_images = image_ops.wrap_img(high_quality_images)
    # cv2.imwrite("high_quality.png", high_quality_images)
    high_quality_images = transform(high_quality_images)
    ref.append(high_quality_images)
    low_quality_images = load.load_h264_training(low_quality_file)
    low_quality_images = image_ops.wrap_img(low_quality_images)
    low_quality_images = transform(low_quality_images)
    composed.append(low_quality_images)
    break

ref = torch.stack(ref).to("cuda:0")
composed = torch.stack(composed).to("cuda:0")
ssim_diff, ssim_loss = cals.mb_ssim(composed, ref)
print(
    f"{dataset}-{low_qp}: 95%: {torch.quantile(ssim_diff.view(-1), 0.95)}, 90%: {torch.quantile(ssim_diff.view(-1), 0.9)}, \
    85%: {torch.quantile(ssim_diff.view(-1), 0.85)}, 80%: {torch.quantile(ssim_diff.view(-1), 0.8)}, \
    75%: {torch.quantile(ssim_diff.view(-1), 0.75)}, 70%: {torch.quantile(ssim_diff.view(-1), 0.7)}, \
    65%: {torch.quantile(ssim_diff.view(-1), 0.65)}, 60%: {torch.quantile(ssim_diff.view(-1), 0.6)}, \
    55%: {torch.quantile(ssim_diff.view(-1), 0.55)}, 50%: {torch.quantile(ssim_diff.view(-1), 0.5)}"
)
ssim_diff = ssim_diff.view(1088 // 16, 1920 // 16).cpu().numpy()
print(f"average ssim diff: {ssim_diff.mean()}")
print(f"ssim diff: {ssim_diff.shape}, ssim loss: {ssim_loss}")
plt.figure(figsize=(120, 90))
colors = ["#0075ce", "#4c6bd1", "#9a59c1", "#e43675", "#e43675"]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
sns.heatmap(ssim_diff, cmap=cmap, cbar=True, annot=True, fmt=".2f")
plt.tight_layout()
plt.savefig(f"ssim_diff-{dataset}-{low_qp}.pdf")
