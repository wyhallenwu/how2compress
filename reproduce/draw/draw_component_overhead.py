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
# plt.rcParams["font.weight"] = "bold"

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

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(7, 6))

# Set global font sizes
plt.rcParams['font.size'] = 30
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
plt.rcParams['legend.fontsize'] = 34

# Bar plot (latency)
x = np.arange(len(latency_keys))
bar_width=0.2
bars = ax1.bar(x, [latency[k] for k in latency_keys], color=blue, alpha=0.85, edgecolor="black", linewidth=1.2, label="Latency (ms)")
# ax1.set_xlabel("Codec", fontsize=12)
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


# Show plot
plt.tight_layout()
plt.savefig("graph/comparison_w_other_codecs.pdf", dpi=2400)

