import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.font_manager as fm
from matplotlib.ticker import LogLocator, ScalarFormatter
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Load the fonts
font_path1 = "/usr/share/fonts/truetype/msttcorefonts/arialbd.ttf"
prop1 = fm.FontProperties(fname=font_path1)
font_path2 = "/usr/share/fonts/truetype/msttcorefonts/arial.ttf"
prop2 = fm.FontProperties(fname=font_path2)
font_path4 = "/usr/share/fonts/truetype/msttcorefonts/times.ttf"
prop4 = fm.FontProperties(fname=font_path4)
font_path5 = "/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf"
prop5 = fm.FontProperties(fname=font_path5)
font_path6 = "/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf"
prop6 = fm.FontProperties(fname=font_path6)
fm.fontManager.addfont(font_path1)
fm.fontManager.addfont(font_path2)
fm.fontManager.addfont(font_path4)
fm.fontManager.addfont(font_path5)
fm.fontManager.addfont(font_path6)
plt.rcParams["font.family"] = [prop1.get_name(), prop2.get_name(), prop5.get_name(), prop6.get_name()]

# Data from the table
data = {
    "Resolution": [
        # 3090 how2compress
        "1080p",
        "900p",
        "720p",
        "480p",
        # orin nano how2compress
        "1080p",
        "900p",
        "720p",
        "480p",
        # agx how2compress
        "1080p",
        "900p",
        "720p",
        "480p",
        # 3090 accmpeg
        "1080p",
        "900p",
        "720p",
        "480p",
        # orin nano accmpeg
        "1080p",
        "900p",
        "720p",
        "480p",
        # accmpeg agx
        "1080p",
        "900p",
        "720p",
        "480p",

    ],
    "Inference Time (ms)": [
        # 3090 how2compress
        10.8,
        9.8,
        9.5,
        9.4,
        # orin nano how2compress
        32,
        31,
        30.3,
        29.5,
        # agx how2compress
        51.7,
        50.9,
        50.4,
        49.8,
        # 3090 accmpeg
        12.3,
        10.4,
        9.5,
        9.2,
        # orin nano accmpeg
        32.7,
        31.6,
        31.1,
        25.2,
        # accmpeg agx
        65.1,
        63.1,
        52.7,
        49.7,
    ],
    "Peak Memory Usage (MB)": [
        # 3090 how2compress
        62,
        48,
        37,
        24,
        # orin nano how2compress
        166,
        151,
        153,
        150,
        # agx how2compress
        174,
        165,
        157,
        148,
        # 3090 accmpeg
        118,
        84,
        58.2,
        25,
        # orin nano accmpeg
        234,
        203,
        180,
        151,
        # accmpeg agx,
        236,
        207,
        184,
        155,


    ],
    "Device": [
        "RTX3090",
        "RTX3090",
        "RTX3090",
        "RTX3090",
        "Jetson Orin Nano",
        "Jetson Orin Nano",
        "Jetson Orin Nano",
        "Jetson Orin Nano",
        "AGX Xavier",
        "AGX Xavier",
        "AGX Xavier",
        "AGX Xavier",
        "RTX3090",
        "RTX3090",
        "RTX3090",
        "RTX3090",
        "Jetson Orin Nano",
        "Jetson Orin Nano",
        "Jetson Orin Nano",
        "Jetson Orin Nano",
        "AGX Xavier",
        "AGX Xavier",
        "AGX Xavier",
        "AGX Xavier",
    ],
    "Method": [
        "How2Compress",
        "How2Compress",
        "How2Compress",
        "How2Compress",
        "How2Compress",
        "How2Compress",
        "How2Compress",
        "How2Compress",
        "How2Compress",
        "How2Compress",
        "How2Compress",
        "How2Compress",
        "AccMPEG",
        "AccMPEG",
        "AccMPEG",
        "AccMPEG",
        "AccMPEG",
        "AccMPEG",
        "AccMPEG",
        "AccMPEG",
        "AccMPEG",
        "AccMPEG",
        "AccMPEG",
        "AccMPEG",
    ],
}

colors = {"How2Compress": "#6A8EC9", "AccMPEG": "#E84446"}
markers = {"1080p": "D", "900p": "o", "720p": "s", "480p": "p"}

# Creating a DataFrame
df = pd.DataFrame(data)

# Create a figure with two subplots
fig, axes = plt.subplots(1, 3, figsize=(7, 6), sharey=True)

# Subplot for Jetson Orin Nano
sns.scatterplot(
    ax=axes[0],
    data=df[df["Device"] == "Jetson Orin Nano"],
    x="Inference Time (ms)",
    y="Peak Memory Usage (MB)",
    hue="Method",
    style="Resolution",
    markers=markers,
    palette=colors,
    linewidth=2,
    edgecolor="black",
    s=200,
    legend=False
)
axes[0].set_xlabel("", fontsize=18, fontproperties=prop5)
axes[0].set_ylabel("Peak GPU Memory Usage (MB)", fontsize=24, fontproperties=prop6)
axes[0].grid(True, color="grey", linewidth=0.3, linestyle="--")

# Subplot for RTX 3090
sns.scatterplot(
    ax=axes[2],
    data=df[df["Device"] == "RTX3090"],
    x="Inference Time (ms)",
    y="Peak Memory Usage (MB)",
    hue="Method",
    style="Resolution",
    markers=markers,
    palette=colors,
    edgecolor="black",
    linewidth=2,
    s=200,
    legend=False,
)
axes[2].set_xlabel("", fontsize=15, fontproperties=prop5)
axes[2].set_ylabel("Peak GPU Memory Usage (MB)", fontsize=24, fontproperties=prop6)
# axes[2].tick_params(axis="both", labelsize=16)
axes[2].grid(True, color="grey", linewidth=0.3, linestyle="--")

# Subplot for AGX Xavier
sns.scatterplot(
    ax=axes[1],
    data=df[df["Device"] == "AGX Xavier"],
    x="Inference Time (ms)",
    y="Peak Memory Usage (MB)",
    hue="Method",
    style="Resolution",
    markers=markers,
    palette=colors,
    edgecolor="black",
    linewidth=2,
    s=200,
    legend=False
)
axes[1].set_xlabel("Inference Time (ms)", fontsize=24, fontproperties=prop6)
axes[1].set_ylabel("Peak GPU Memory Usage (MB)", fontsize=24, fontproperties=prop6)
axes[1].grid(True, color="grey", linewidth=0.3, linestyle="--")
# axes[1].tick_params(axis="both", labelsize=16)
# axes[0].tick_params(axis="both", labelsize=16)

# Set titles AFTER creating the plots to ensure font properties are applied
axes[0].set_title("Jetson Orin Nano", fontsize=24, fontproperties=prop6)
axes[1].set_title("AGX Xavier", fontsize=24, fontproperties=prop6)
axes[2].set_title("RTX 3090", fontsize=24, fontproperties=prop6)

# Apply font properties to all tick labels
for ax in axes:
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(prop5)
        tick.set_fontsize(16)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(prop5)
        tick.set_fontsize(16)

# Set titles with explicit font properties - do this LAST to avoid overrides
titles = ["Jetson Orin Nano", "AGX Xavier", "RTX 3090"]
for ax, title_text in zip(axes, titles):
    # First set the title
    ax.set_title(title_text)
    # Then get the title object and modify it directly
    title_obj = ax.title
    title_obj.set_fontproperties(prop6)
    title_obj.set_fontsize(16)
    # title_obj.set_fontweight('bold')
# Rotate y-axis tick labels by 90 degrees for all subplots
for ax in axes:
    ax.tick_params(axis='y', rotation=90)

# Text annotation with prop5 font
fig.text(
    0.15,  
    0.18,  
    "Computational Overhead (1080p):\nMAX FLOPs 7.26G\nMAX MACs 3.58G",
    fontsize=16, color="black",
    ha="left", va="center", fontproperties=prop6,
    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5")
)

# Custom legend handles
method_handles = [Patch(facecolor=colors[m], label=m, edgecolor='black') for m in colors]
resolution_handles = [
    Line2D([0], [0], marker=markers[r], color='black', linestyle='None',
           markersize=10, label=r, markerfacecolor='white', markeredgewidth=2)
    for r in markers
]

# Method legend with prop5 font
method_legend = fig.legend(
    handles=method_handles,
    title="Method",
    title_fontsize=16,
    loc='lower left',
    bbox_to_anchor=(0.12, 0.25),
    fontsize=16,
    frameon=True,
    handlelength=1.8,
    labelspacing=0.6,
    borderpad=0.5,
    prop=prop5
)

# Set title font properties for method legend
method_legend.get_title().set_fontproperties(prop6)
method_legend.get_title().set_fontsize(16)
for text in method_legend.get_texts():
    text.set_fontproperties(prop6)
    text.set_fontsize(14)

# Resolution legend with prop5 font
resolution_legend = axes[2].legend(
    handles=resolution_handles,
    title="Resolution",
    title_fontsize=16,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.95),
    fontsize=16,
    frameon=True,
    handlelength=1.2,
    labelspacing=0.5,
    borderpad=0.5,
    prop=prop5
)

# Set title font properties for resolution legend
resolution_legend.get_title().set_fontproperties(prop6)
resolution_legend.get_title().set_fontsize(16)
for text in resolution_legend.get_texts():
    text.set_fontproperties(prop6)
    text.set_fontsize(14)

plt.grid(True, color="grey", linewidth=0.3, linestyle="--")
plt.savefig("graph/inference_time_eval.pdf", dpi=2400, bbox_inches="tight")