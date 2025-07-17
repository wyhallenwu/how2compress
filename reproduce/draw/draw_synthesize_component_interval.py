import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.font_manager as fm
from matplotlib.ticker import LogLocator, ScalarFormatter
import numpy as np

# Load the correct font
font_path = "/usr/share/fonts/truetype/msttcorefonts/arialbd.ttf"
prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
font1_path = "/usr/share/fonts/truetype/msttcorefonts/Verdana_Bold_Italic.ttf"
fm.fontManager.addfont(font1_path)
verdanabi = fm.FontProperties(fname=font1_path)

plt.rcParams["font.family"] = prop.get_name()
plt.rcParams["font.weight"] = "bold"

# Define colors
red = "#D77071"  
blue = "#6888F5"  

# Data for first plot
latency = {
    "H.264/wo": 16.29,  
    "H.265": 37.02,    
    "VP9": 260.78,    
    "VP9/mmt": 129.5, 
    "H.264/w": 22.57,  
}

bitrate = {
    "H.264/wo": 2.767,
    "H.265": 2.220,
    "VP9": 7.993,
    "VP9/mmt": 8.272,
    "H.264/w": 2.023,
}

latency_keys = list(latency.keys())
bitrate_keys = list(bitrate.keys())

# Data for second plot
interval_performance_on_mot = {
    0: 1,
    1: 0.99,
    5: 0.95,
    10: 0.92,
    30: 0.88
}

interval_performance_on_aicity = {
    0: 1,
    1: 1,
    5: 0.93,
    10: 0.91,
    30: 0.81
}

df_mot = pd.DataFrame({"Interval": list(interval_performance_on_mot.keys()),
                        "Accuracy": list(interval_performance_on_mot.values()),
                        "Category": "MOT"})

df_aicity = pd.DataFrame({"Interval": list(interval_performance_on_aicity.keys()),
                           "Accuracy": list(interval_performance_on_aicity.values()),
                           "Category": "AICITY"})

df = pd.concat([df_mot, df_aicity])
palette = {"MOT": red, "AICITY": blue}

# Create a single figure with two subplots
fig, axes = plt.subplots(nrows=2, figsize=(8,7), gridspec_kw={'height_ratios': [4, 3]})

# First subplot (Latency and Bitrate)
ax1 = axes[0]
x = np.arange(len(latency_keys))
bar_width = 0.4

bars = ax1.bar(x, [latency[k] for k in latency_keys], color=blue, alpha=0.85, edgecolor="black", linewidth=1.2, label="Latency (ms)")
ax1.set_ylabel("Latency (ms)", color=blue, fontsize=20)
ax1.tick_params(axis='y', labelcolor=blue)
ax1.set_xticks(x)
ax1.set_xticklabels(latency_keys)

ax1.bar(x, [latency[k] for k in latency_keys], facecolor='none', hatch='/', edgecolor="white", linewidth=1.2)
ax1.set_yticks([0, 100, 200, 320])

for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=14, color='black')

ax2 = ax1.twinx()
ax2.tick_params(axis='y', labelsize=16)
bitrate_x = np.arange(len(bitrate_keys))
ax2.plot(bitrate_x, [bitrate[k] for k in bitrate_keys], marker='o', color=red, linestyle='-', alpha=0.85, label="Bitrate (kbps)", linewidth=2.5)
ax2.set_ylabel("Bitrate (Mbps)", color=red, fontsize=18)
ax2.set_yticks([0, 2, 4, 6, 8])
ax2.tick_params(axis='y', labelcolor=red)
ax1.set_xticklabels(latency_keys, rotation=0, fontsize=18)

# Second subplot (Interval vs Accuracy)
sns.lineplot(ax=axes[1], data=df, x="Interval", y="Accuracy", hue="Category", marker="o", linewidth=2.5, palette=palette, alpha=0.85)

axes[1].set_xlabel("Interval", fontsize=20, fontweight="bold")
axes[1].set_ylabel("Accuracy", fontsize=20, fontweight="bold")
axes[1].set_xticks(df["Interval"].unique())
axes[1].set_yticks([0.8, 1], labels=["0.8x", "1x"])
axes[1].set_ylim(0.6, 1.1)
axes[1].grid(True, linestyle="--", alpha=0.6)
axes[1].legend(title="", fontsize=14)

axes[0].tick_params(axis="both", labelsize=18)
axes[1].tick_params(axis="both", labelsize=18)
# plt.tight_layout()
plt.savefig("graph/component-interval.pdf", dpi=2400)
