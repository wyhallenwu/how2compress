import csv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.colors import ListedColormap

filename1 = "/how2compress/data/accmpeg_gt/MOT17-04-0.75q.txt"
filename2 = "/how2compress/data/accmpeg_gt/detr-MOT17-04-0.7q.txt"
# filename = "/how2compress/results/decisions30-45-MOT17-04-f7.txt"
# frame = "/how2compress/data/MOT17Det/train/MOT17-04/img1/000001.jpg"

data1 = np.loadtxt(filename1)
data2 = np.loadtxt(filename2)
print(data1.shape)
plt.figure(figsize=(4, 3))
data1 = data1[0].reshape(68, 120)
data2 = data2[0].reshape(68, 120)
binary_mask = (data1 == data2).astype(int)
custom_cmap = ListedColormap(["#E84446", "#6A8EC9"])


heatmap = sns.heatmap(
    binary_mask,
    cmap="coolwarm",
    cbar=False,
    annot=False,
    # alpha=1,
    # zorder=2,
    linewidths=0,
    linecolor=None,
)
# Access the colorbar object
# colorbar = heatmap.collections[0].colorbar
# bg = Image.open(frame)
# bg = bg.resize((120, 68))

# plt.imshow(bg, aspect="auto", zorder=1)


# # Set custom ticks for the colorbar
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
plt.savefig("graph/target-diff-1704.pdf")
# plt.close()
