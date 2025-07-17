import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.font_manager as fm
import numpy as np
from matplotlib.ticker import LogLocator, ScalarFormatter

# Load the correct font
font_path = "/usr/share/fonts/truetype/msttcorefonts/arialbd.ttf"
prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
# font1_path = "/usr/share/fonts/truetype/msttcorefonts/Verdana_Bold_Italic.ttf"
# fm.fontManager.addfont(font1_path)
# verdanabi= fm.FontProperties(fname=font1_path)

plt.rcParams["font.family"] = prop.get_name()
plt.rcParams["font.weight"] = "bold"

red = "#D77071"  
blue = "#6888F5"  

# Prepare data
data = {
    "Mask": [30, 35, 40, 45, 50] * 2,
    "Accuracy": [1.00, 0.96, 0.92, 0.85, 0.75,   # RoI
                 1.00, 0.96, 0.94, 0.92, 0.89],  # BG
    "Type": ["RoI"] * 5 + ["BG"] * 5
}
df = pd.DataFrame(data)

# Create base plot
plt.figure(figsize=(7, 5))
ax = sns.barplot(
    data=df,
    x="Mask", y="Accuracy", hue="Type",
    palette={"RoI": red, "BG": blue},
    edgecolor="black", alpha=0.85
)

# X locations
n_masks = 5
x_labels = sorted(df["Mask"].unique())
x = np.arange(n_masks)
bar_width = 0.4

# Get values for overlay
roi_acc = df[df["Type"] == "RoI"]["Accuracy"].values
bg_acc = df[df["Type"] == "BG"]["Accuracy"].values

# Overlay white hatch
ax.bar(x - bar_width/2, roi_acc, width=bar_width, facecolor="none", edgecolor="white",
       linewidth=1.5, hatch='\\')
ax.bar(x + bar_width/2, bg_acc, width=bar_width, facecolor="none", edgecolor="white",
       linewidth=1.5, hatch='/')

# === Label values on bars ===
# Loop through seaborn bars (10 in total: 5 RoI + 5 BG)
for i, bar in enumerate(ax.patches):
    height = bar.get_height()
    x_pos = bar.get_x() + bar.get_width() / 2
    print(i, height)
    if i not in [0, 1, 5, 12, 17, 6, 13, 18]:
        ax.text(x_pos, height + 0.01, f"{height:.2f}×",
            ha='center', va='bottom', fontsize=16, fontweight='bold')
    elif i in [0, 1]:
        ax.text(bar.get_x()+bar.get_width(), height + 0.01, f"{height:.2f}×",
            ha='center', va='bottom', fontsize=16, fontweight='bold')
    

# Axis formatting
ax.set_ylim(0.7, 1.05)  # Set y-axis starting from 0.6
ax.set_ylabel("Relative Accuracy", fontsize=18, weight='bold')
ax.set_xlabel("QP", fontsize=18, weight='bold')
ax.set_xticks(x)
ax.set_xticklabels([str(v) for v in x_labels], fontsize=16, weight='bold')
ax.set_yticks([0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels(["0.7", "0.8", "0.9", "1.0"], fontsize=16, weight='bold', rotation=45)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f×'))
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

# Custom legend
bar_legend_patches = [
    mpatches.Patch(facecolor=red, edgecolor="white", label="RoI", hatch='\\'),
    mpatches.Patch(facecolor=blue, edgecolor="white", label="BG", hatch='/')
]
ax.legend(handles=bar_legend_patches, title="", fontsize=16, loc="upper right", frameon=False)

plt.savefig("graph/effect_roi.pdf", dpi=2400)