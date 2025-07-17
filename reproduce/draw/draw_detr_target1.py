import csv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.colors import ListedColormap

# filename = "/how2compress/data/accmpeg_gt/MOT17-04-0.75q.txt"
filename = "/how2compress/data/accmpeg_gt/detr-MOT17-04-0.7q.txt"
# filename = "/how2compress/results/decisions30-45-MOT17-04-f7.txt"
# frame = "/how2compress/data/MOT17Det/train/MOT17-04/img1/000001.jpg"

data = np.loadtxt(filename)
print(data.shape)
plt.figure(figsize=(4, 3))
data = data[0].reshape(68, 120)
custom_cmap = ListedColormap(["#6A8EC9", "#E84446"])
heatmap = sns.heatmap(
    data,
    cmap="coolwarm",
    cbar=False,
    annot=False,
    # alpha=1,
    # zorder=2,
    linewidths=0,
)
# Access the colorbar object
colorbar = heatmap.collections[0].colorbar
# bg = Image.open(frame)
# bg = bg.resize((120, 68))

# plt.imshow(bg, aspect="auto", zorder=1)


# Set custom ticks for the colorbar
# custom_ticks = [0, 1]  # Define the tick positions
# colorbar.set_ticks(custom_ticks)  # Apply the custom ticks
# colorbar.set_ticklabels(["", ""])  # Apply custom labels
# colorbar.ax.tick_params(left=False, right=False, labelleft=False, labelright=True)
# # Optional: Customize the colorbar label and tick size
# colorbar.set_label("Emphasis Level", fontsize=14)
# colorbar.ax.yaxis.set_tick_params(labelsize=10)
plt.axis("off")
plt.tight_layout()
plt.gcf().set_dpi(2400)
plt.savefig("graph/target-yolo-1704.pdf")
plt.close()
