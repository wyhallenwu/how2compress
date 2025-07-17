import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.font_manager as fm
import numpy as np

# Load the correct font
font_path1 = "/usr/share/fonts/truetype/msttcorefonts/arialbd.ttf"
prop1 = fm.FontProperties(fname=font_path1)
font_path2 = "/usr/share/fonts/truetype/msttcorefonts/arial.ttf"
prop2 = fm.FontProperties(fname=font_path2)
font_path4 = "/usr/share/fonts/truetype/msttcorefonts/times.ttf"
prop4 = fm.FontProperties(fname=font_path4)
fm.fontManager.addfont(font_path1)
fm.fontManager.addfont(font_path2)
fm.fontManager.addfont(font_path4)
plt.rcParams["font.family"] = [prop1.get_name(), prop2.get_name()]

red = "#D77071"
blue = "#6888F5"

# Prepare data
data = {
    "Mask": [30, 35, 40, 45, 50] * 2,
    "Accuracy": [1.00, 0.96, 0.92, 0.85, 0.75,
                 1.00, 0.96, 0.94, 0.92, 0.89],
    "Type": ["RoI"] * 5 + ["BG"] * 5
}
df = pd.DataFrame(data)

# Create base plot
plt.figure(figsize=(6, 6))  # Keep square aspect ratio
ax = sns.barplot(
    data=df,
    y="Mask", x="Accuracy", hue="Type",
    palette={"RoI": red, "BG": blue},
    edgecolor="black", alpha=0.85,
    orient="h"  # horizontal orientation
)

# Y locations
n_masks = 5
y_labels = sorted(df["Mask"].unique())
y = np.arange(n_masks)
bar_height = 0.4

# Get values for overlay
roi_acc = df[df["Type"] == "RoI"]["Accuracy"].values
bg_acc = df[df["Type"] == "BG"]["Accuracy"].values

# Overlay white hatch
ax.barh(y - bar_height/2, roi_acc, height=bar_height, facecolor="none", edgecolor="white",
        linewidth=1.5, hatch='\\')
ax.barh(y + bar_height/2, bg_acc, height=bar_height, facecolor="none", edgecolor="white",
        linewidth=1.5, hatch='/')

# Label values on bars
for i, bar in enumerate(ax.patches):
    width = bar.get_width()
    y_pos = bar.get_y() + bar.get_height() / 2
    if i not in [0, 1, 5, 12, 17, 6, 13, 18]:
        ax.text(width + 0.002, y_pos, f"{width:.2f}×",
                va='center', ha='left', fontsize=24, fontproperties=prop4)
    elif i in [0, 1]:
        ax.text(width + 0.002, y_pos, f"{width:.2f}×",
                va='center', ha='left', fontsize=24, fontproperties=prop4)

# Axis formatting
ax.set_xlim(0.7, 1.05)
ax.set_xlabel("Relative Accuracy", fontproperties=prop4, fontsize=24)
ax.set_ylabel("QP", fontproperties=prop4, fontsize=24)
ax.set_yticks(y)
ax.set_yticklabels([str(v) for v in y_labels], fontproperties=prop4, fontsize=22, rotation=90)
ax.set_xticks([0.7, 0.8, 0.9, 1.0])
ax.set_xticklabels(["0.7", "0.8", "0.9", "1.0"], fontproperties=prop4, fontsize=22)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f×'))
ax.grid(axis="x", linestyle="--", linewidth=0.5, color='lightgray', alpha=0.9)

# Set spine colors
ax.spines['top'].set_color('lightgray')
ax.spines['right'].set_color('lightgray')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
# Set spine widths
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

# Custom legend
bar_legend_patches = [
    mpatches.Patch(facecolor=red, edgecolor="white", label="RoI", hatch='\\'),
    mpatches.Patch(facecolor=blue, edgecolor="white", label="BG", hatch='/')
]
ax.legend(handles=bar_legend_patches, title="", fontsize=26, loc="lower right", frameon=False)

plt.tight_layout()
plt.savefig("graph/effect_roi_horizontal.pdf", dpi=2400)
