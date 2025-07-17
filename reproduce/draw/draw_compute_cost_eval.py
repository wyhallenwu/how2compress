import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.font_manager as fm
from matplotlib.ticker import LogLocator, ScalarFormatter

# Load the correct font
font_path = "/usr/share/fonts/truetype/msttcorefonts/arialbd.ttf"
prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
font1_path = "/usr/share/fonts/truetype/msttcorefonts/Verdana_Bold_Italic.ttf"
fm.fontManager.addfont(font1_path)
verdanabi= fm.FontProperties(fname=font1_path)

plt.rcParams["font.family"] = prop.get_name()
plt.rcParams["font.weight"] = "bold"

# Data from the tables
device_data = {
    "Device": ["Orin Nano (4GB)", "Orin Nano (8GB)", "Xavier NX", "AGX Xavier", "RTX 3090"],
    "GFLOPS": [640, 1280, 844, 1410, 35580]
}

resolution_data = {
    "Resolution": ["1080p", "900p", "720p", "480p"],
    "GFLOPS": [7.256, 3.023, 1.934, 0.868],
    "GMACs": [3.582, 1.493, 0.955, 0.429]
}

red = "#D77071"  # Now GFLOPS color
blue = "#6888F5"  # Now GMACs color

# Convert to DataFrame
device_df = pd.DataFrame(device_data)
resolution_df = pd.DataFrame(resolution_data)

# Convert GFLOPS to GMACs (1 GFLOPS ≈ 0.5 GMACs)
device_df["GMACs"] = device_df["GFLOPS"] * 0.5

# Set up the figure and primary axis
fig, ax1 = plt.subplots(figsize=(7, 6))

# Create secondary axis
ax2 = ax1.twinx()

# Set width for grouped bar plots
bar_width = 0.3
x = range(len(resolution_df["Resolution"]))

# **Now GFLOPS is on ax1 (Left)**
bars1 = ax1.bar([i - bar_width/2 for i in x], resolution_df["GFLOPS"], width=bar_width, 
                color=red, alpha=0.85, label="Model GFLOPS",
                edgecolor="black", linewidth=1.2)

# **Now GMACs is on ax2 (Right)**
bars2 = ax2.bar([i + bar_width/2 for i in x], resolution_df["GMACs"], width=bar_width, 
                color=blue, alpha=0.85, label="Model GMACs",
                edgecolor="black", linewidth=1.2)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height * 1.1, f'{height:.2f}', ha='center', fontsize=14, color=red, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height * 1.1, f'{height:.2f}', ha='center', fontsize=14, color=blue, fontweight='bold')


# Overlay bars with white hatch pattern
ax1.bar([i - bar_width/2 for i in x], resolution_df["GFLOPS"], width=bar_width, 
        facecolor="none", edgecolor="white", linewidth=1.5, hatch='\\')  # White hatch for GFLOPS

ax2.bar([i + bar_width/2 for i in x], resolution_df["GMACs"], width=bar_width, 
        facecolor="none", edgecolor="white", linewidth=1.5, hatch='/')  # White hatch for GMACs

# Update legend patches
bar_legend_patches = [
    mpatches.Patch(facecolor=red, edgecolor="black", label="Model GFLOPS", hatch='\\'),
    mpatches.Patch(facecolor=blue, edgecolor="black", label="Model GMACs", hatch='/')
]

# Draw upper bound lines for devices (on GFLOPS scale, which is now ax1)
colors = ["#0A87D7", "#6F5B9B", "#81AA2A", "#AEA3CD", "#B5711E"]
device_legends = []
for idx, (_, row) in enumerate(device_df.iterrows()):
    ax1.axhline(y=row["GFLOPS"], linestyle="--", color=colors[idx], alpha=0.7, linewidth=4)
    device_legends.append(mlines.Line2D([], [], color=colors[idx], linestyle="--", linewidth=3, label=row["Device"]))

# ax1.text(
#     2,  
#     120,  
#     "Lightweight & Edge-Friendly",  # Text to display
#     fontsize=20, fontweight="bold", color="red",
#     ha="center", va="center", fontproperties=verdanabi,
#     bbox=dict(facecolor="white", edgecolor="white", boxstyle="round,pad=0.5")
# )

orin_4gb_gflops = device_df.loc[device_df["Device"] == "Orin Nano (4GB)", "GFLOPS"].values[0]
ax1.text(
    x=0.4, y=orin_4gb_gflops * 1.05,  # Position slightly above the line
    s="640 GFLOPS",
    fontsize=14, fontweight="bold", color="#0A87D7",
    ha="left", va="bottom",
    bbox=dict(facecolor="white", edgecolor="#0A87D7", boxstyle="round,pad=0.3")
)

# Set x-axis ticks and labels
ax1.set_xticks(x)
ax1.set_xticklabels(resolution_df["Resolution"], fontsize=18, fontweight="bold")

# **Update Axis Labels**
ax1.set_xlabel("Resolution", fontsize=20, fontweight="bold")
ax1.set_ylabel("GFLOPs(log)", fontsize=18, fontweight="bold", color=red)  # Left y-axis now GFLOPS
ax2.set_ylabel("GMACs(log)", fontsize=18, fontweight="bold", color=blue)  # Right y-axis now GMACs

# Adjust tick parameters
ax1.tick_params(axis='y', colors=red, labelsize=16)  # GFLOPS in red
ax2.tick_params(axis='y', colors=blue, labelsize=16)  # GMACs in blue

# **Set Log Scale for Both Axes**
ax1.set_yscale("log")  # GFLOPS now follows log scale
ax2.set_yscale("log")  # GMACs now follows log scale

# **Logarithmic tick settings**
ax1.yaxis.set_major_locator(LogLocator(base=10.0, subs='auto', numticks=10))
ax1.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[0.2, 0.5, 0.8], numticks=10))

ax2.yaxis.set_major_locator(LogLocator(base=10.0, subs='auto', numticks=10))
ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[0.2, 0.5, 0.8], numticks=10))

# **Apply scientific notation format for tick labels**
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0e'))  # GFLOPS (Left)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0e'))  # GMACs (Right)

# Hide minor tick labels to avoid clutter
ax1.yaxis.set_minor_formatter(mticker.NullFormatter())
ax2.yaxis.set_minor_formatter(mticker.NullFormatter())

# Set specific y-ticks for better spacing
ax1.set_yticks([1e0, 1e1, 1e2, 1e3, 1e4])  # GFLOPS ticks
ax2.set_yticks([1e-1, 1e0, 1e1, 1e2])  # GMACs ticks

# Combine legends
all_legends = bar_legend_patches + device_legends

# Set single legend at upper right with adjusted placement
ax1.legend(handles=all_legends, loc="upper right", fontsize=12, framealpha=1)

# Add grid for better readability
ax1.grid(axis='y', linestyle='--', alpha=0.6)

# Save the figure with high resolution
plt.savefig("graph/compute-cost-eval.pdf", dpi=2400, bbox_inches="tight")
