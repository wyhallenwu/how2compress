import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.font_manager as fm
from matplotlib.ticker import LogLocator, ScalarFormatter
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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

# Create figure with two subplots side by side
fig = plt.figure(figsize=(14, 6))

# ==================== LEFT SUBPLOT: Component Overhead ====================
ax1 = plt.subplot(1, 2, 1)

latency = {
    "H.264 w/o": 16.29,  # 5.685/349
    "H.264 w/\n(Ours)": 22.57,   # 7.877/349
    "H.265": 37.02,      # 12.92/349
    "AccMPEG": 27.57,    # 9.621/349
    "VP9": 260.78,      # 91.015/349,
    # "vp9/mmt": 129.5, # 45.2/349
    "H.266": 1077,         # 375.9 / 349  
    "AV1": 1589          # 554.6 / 349
    
}

bitrate = {
    "H.264 w/o": 2.767,
    "H.264 w/\n(Ours)": 2.023,
    "H.265": 2.220,
    "AccMPEG": 3.354,
    "VP9": 7.993,
    # "vp9/mmt": 8272,
    "H.266": 2.415,
    "AV1": 4.772
    
}

red = "#D77071"  
blue = "#6888F5"  

# Align keys for consistent ordering
latency_keys = list(latency.keys())
bitrate_keys = list(bitrate.keys())

# Bar plot (latency)
x = np.arange(len(latency_keys))
bar_width=0.2
bars = ax1.bar(x, [latency[k] for k in latency_keys], color=blue, alpha=0.85, edgecolor="black", linewidth=1.2, label="Latency (ms)")
ax1.set_ylabel("Latency (x100ms)", color=blue, fontsize=24, fontproperties=prop6)
ax1.tick_params(axis='y', labelcolor=blue, labelsize=24)
ax1.set_ylim(0, 320)
ax1.set_xticks(x)
ax1.set_xticklabels(latency_keys, fontsize=30, fontproperties=prop5)
ylim = 320
ax1.bar(x, [min(latency[k], ylim) for k in latency_keys], facecolor='none', hatch='/', edgecolor="white", linewidth=1.2)
ax1.set_yticks([0, 100, 200, 300])
ax1.set_yticklabels(['0x', '1x', '2x', '3x'], fontsize=30, rotation=90, fontproperties=prop5)

for i, bar in enumerate(bars):
    if i > 3:
        continue
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', 
             ha='center', va='bottom', fontsize=20, color=blue, zorder=6, fontproperties=prop5)

last_two_indices = x[-2:]  # get the x positions of VVC and AV1
mask_x_start = last_two_indices[0] - bar_width * 2.5
mask_width = bar_width * 10

# Add semi-transparent rectangle
mask = plt.Rectangle(
    (mask_x_start, 0),         # Bottom-left corner (x, y)
    mask_width,                # Width to cover last two bars
    ylim,                      # Height (up to the top of y-axis)
    color='grey',
    alpha=0.55,
    zorder=5
)
ax1.add_patch(mask)

# Add text annotation
ax1.text(
    mask_x_start + mask_width / 2,     # x: center of the rectangle
    ylim * 0.9,                          # y: middle of the visible y-axis
    "H.266 & AV1\n incur prohibited\nlatency on edge", 
    ha='center', va='center', 
    fontsize=17, 
    color='red', zorder=7, fontproperties=prop5
)

# Create a second y-axis for bitrate
ax2 = ax1.twinx()
bitrate_x = np.arange(len(bitrate_keys))
ax2.plot(bitrate_x, [bitrate[k] for k in bitrate_keys], marker='o', color=red, linestyle='-', alpha=0.85, label="Bitrate (Mbps)", linewidth=2.5)
ax2.set_ylabel("Bitrate (Mbps)", color=red, fontsize=24, fontproperties=prop6)
ax2.tick_params(axis='y', labelcolor=red, labelsize=24)

# Align x-axis labels
ax1.set_xticks(x)
ax1.set_xticklabels(latency_keys, rotation=30, fontsize=30, fontproperties=prop5)
# Apply font properties to tick labels explicitly
for tick in ax1.get_xticklabels():
    tick.set_fontproperties(prop6)
    tick.set_fontsize(15)

for tick in ax1.get_yticklabels():
    tick.set_fontproperties(prop5)
    tick.set_fontsize(20)

for tick in ax2.get_yticklabels():
    tick.set_fontproperties(prop5)
    tick.set_fontsize(20)

ax1.grid(True, linestyle="--")
xtick_labels = ax1.get_xticklabels()
xtick_labels[1].set_color('red') 

# Custom legend handles
latency_patch = Patch(facecolor=blue, edgecolor="black", label="Latency (ms)", linewidth=1.2)
bitrate_line = Line2D([0], [0], color=red, marker='o', label="Bitrate (Mbps)", linewidth=2.5)

# Add legend
legend = ax1.legend(
    handles=[latency_patch, bitrate_line],
    loc='upper left',  # Change location as needed (e.g., 'upper right', 'lower center')
    fontsize=24,
    frameon=True,
    edgecolor="black",
    prop=prop5
)

# Explicitly set font properties for legend text
for text in legend.get_texts():
    text.set_fontproperties(prop6)
    text.set_fontsize(24)

# ==================== RIGHT SUBPLOTS: Inference Time ====================
# Create 3 subplots for inference time evaluation
ax_orin = plt.subplot(1, 4, 2)  # Jetson Orin Nano
ax_agx = plt.subplot(1, 4, 3)   # AGX Xavier  
ax_rtx = plt.subplot(1, 4, 4)   # RTX 3090

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

# Subplot for Jetson Orin Nano
sns.scatterplot(
    ax=ax_orin,
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
ax_orin.set_title("Jetson Orin Nano", fontsize=24, fontproperties=prop6)
ax_orin.set_xlabel("", fontsize=18, fontproperties=prop5)
ax_orin.set_ylabel("Peak GPU Memory Usage (MB)", fontsize=24, fontproperties=prop6)
ax_orin.grid(True, color="grey", linewidth=0.3, linestyle="--")

# Subplot for AGX Xavier
sns.scatterplot(
    ax=ax_agx,
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
ax_agx.set_title("AGX Xavier", fontsize=24, fontproperties=prop6)
ax_agx.set_xlabel("Inference Time (ms)", fontsize=24, fontproperties=prop6)
ax_agx.set_ylabel("Peak GPU Memory Usage (MB)", fontsize=24, fontproperties=prop6)
ax_agx.grid(True, color="grey", linewidth=0.3, linestyle="--")

# Subplot for RTX 3090
sns.scatterplot(
    ax=ax_rtx,
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
ax_rtx.set_title("RTX 3090", fontsize=24, fontproperties=prop6)
ax_rtx.set_xlabel("", fontsize=15, fontproperties=prop5)
ax_rtx.set_ylabel("Peak GPU Memory Usage (MB)", fontsize=24, fontproperties=prop6)
ax_rtx.tick_params(axis="both", labelsize=16)
ax_rtx.grid(True, color="grey", linewidth=0.3, linestyle="--")

# Apply font properties to all tick labels
for ax in [ax_orin, ax_agx, ax_rtx]:
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(prop5)
        tick.set_fontsize(16)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(prop5)
        tick.set_fontsize(16)

# Text annotation with prop5 font
fig.text(
    0.85,  
    0.2,  
    "Computational Overhead (1080p):\nMAX FLOPs 7.26G\nMAX MACs 3.72G",
    fontsize=12, fontweight="bold", color="black",
    ha="left", va="center", fontproperties=prop5,
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
    bbox_to_anchor=(0.75, 0.25),
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

# Resolution legend with prop5 font
resolution_legend = ax_rtx.legend(
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

plt.grid(True, color="grey", linewidth=0.3, linestyle="--")
plt.tight_layout()
plt.savefig("graph/merged_eval.pdf", dpi=2400, bbox_inches="tight") 