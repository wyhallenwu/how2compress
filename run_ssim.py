import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils import cals, image_ops

raw_frame = "/how2compress/data/MOT17Det/train/MOT17-04/img1/000001.jpg"
compressed_frame1 = "/how2compress/graph/000001-accmpeg.png"
compressed_frame2 = "/how2compress/graph/000001-ours.png"

def draw(raw_frame, compressed_frame, ax):
    raw_frame = cv2.imread(raw_frame)
    compressed_frame = cv2.imread(compressed_frame)
    assert raw_frame.shape == compressed_frame.shape

    raw_frame = image_ops.wrap_img(raw_frame)
    compressed_frame = image_ops.wrap_img(compressed_frame)

    raw_frame = image_ops.vit_transform_fn()(raw_frame)
    compressed_frame = image_ops.vit_transform_fn()(compressed_frame)

    raw_frame = raw_frame.unsqueeze(0)
    compressed_frame = compressed_frame.unsqueeze(0)

    ssim, v = cals.mb_ssim(raw_frame, compressed_frame)
    print(v)
    print(ssim.mean())
    ssim = ssim.cpu().numpy().reshape(1088 // 16, 1920 // 16)
    # plt.figure(figsize=(4, 3))
    ax.axis("off")
    heatmap = sns.heatmap(ssim, cmap="coolwarm", cbar=False, annot=False, ax=ax)
    # colorbar = heatmap.collections[0].colorbar
    # custom_ticks = [0, 1]  # Define the tick positions
    # colorbar.set_ticks([])  # Apply the custom ticks
    # # colorbar.set_ticklabels(["", "", "", "", ""])  # Apply custom labels
    # colorbar.set_label("SSIM", fontsize=14)
    # colorbar.ax.yaxis.set_tick_params(labelsize=4)
    # plt.tight_layout()
    # plt.savefig("graph/ssim.pdf")

if __name__ == "__main__":
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams["font.weight"] = "bold"
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 1.8), sharex=True, sharey=True)
    draw(raw_frame, compressed_frame1, axes[0])
    draw(raw_frame, compressed_frame2, axes[1])
    cbar_ax = fig.add_axes([0.9, 0.2, 0.02, 0.55])  # Adjust position and size as needed
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([])
    cbar.set_ticklabels([])
    cbar.set_label("SSIM", fontsize=12, weight="bold")
    # plt.subplots_adjust(wspace=0)
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to accommodate colorbar
    plt.savefig("graph/ssim-all.pdf", dpi=1200, bbox_inches='tight')