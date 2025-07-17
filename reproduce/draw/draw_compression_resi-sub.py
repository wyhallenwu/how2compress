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
from scipy.stats import gaussian_kde
from scipy.interpolate import make_interp_spline


# Load the correct font
font_path1 = "/usr/share/fonts/truetype/msttcorefonts/arialbd.ttf"
prop1 = fm.FontProperties(fname=font_path1)
font_path2 = "/usr/share/fonts/truetype/msttcorefonts/arial.ttf"
prop2 = fm.FontProperties(fname=font_path2)
fm.fontManager.addfont(font_path1)
fm.fontManager.addfont(font_path2)
bd = fm.FontProperties(fname=font_path1)
regular = fm.FontProperties(fname=font_path2)
font_path3 = "/usr/share/fonts/truetype/msttcorefonts/Verdana_Bold_Italic.ttf"
fm.fontManager.addfont(font_path3)
verdanabi= fm.FontProperties(fname=font_path3)

font_path4 = "/usr/share/fonts/truetype/msttcorefonts/times.ttf"
fm.fontManager.addfont(font_path4)
prop4 = fm.FontProperties(fname=font_path4)

plt.rcParams["font.family"] = [prop1.get_name(), prop2.get_name()]
# plt.rcParams["font.weight"] = "bold"

#91CAE8, #F48892
c1 = "#91CAE8"
c2 = "#F48892"

# Original data
qp45_raw = {0.87: 0.25, 0.93: 0.5, 0.97: 0.25}
qp35_raw = {0.94: 0.25, 0.96: 0.5, 0.98: 0.25}

# Add (0,0) and (1,0)
def extend_with_zeros(data_dict):
    extended = {0.0: 0.0}
    extended.update(data_dict)
    extended[1.0] = 0.0
    return extended

qp45 = extend_with_zeros(qp45_raw)
qp35 = extend_with_zeros(qp35_raw)

# Function to plot smoothed PDF
def plot_pdf(ax, data_dict, label, color, linestyle='-'):
    x = np.array(sorted(data_dict.keys()))
    y = np.array([data_dict[k] for k in x])
    x_smooth = np.linspace(0, 1, 300)
    spline = make_interp_spline(x, y, k=3)
    y_smooth = spline(x_smooth)
    # Add light fill under the line
    ax.fill_between(x_smooth, y_smooth, alpha=0.2, color=color)
    ax.plot(x_smooth, y_smooth, label=label, color=color, linestyle=linestyle, linewidth=4)
    for x_val, y_val in data_dict.items():
        ax.vlines(x_val, 0, y_val, color=color, linestyle='--', linewidth=1.5)

# Create vertically stacked subplots
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 6), sharex=True)

# Plot QP 45
plot_pdf(axs[0], qp45, label="QP 45", color=c1)
axs[0].legend(fontsize=26, loc='upper left', frameon=True)
axs[0].set_ylim(0, 0.6)
axs[0].grid(True, linestyle='--', linewidth=0.5, color='lightgray', alpha=0.9)
# Set spine colors for first subplot
axs[0].spines['top'].set_color('lightgray')
axs[0].spines['right'].set_color('lightgray')
axs[0].spines['bottom'].set_color('black')
axs[0].spines['left'].set_color('black')
# Set spine widths
axs[0].spines['bottom'].set_linewidth(1.5)
axs[0].spines['left'].set_linewidth(1.5)
axs[0].set_yticks([0.2, 0.4, 0.6])
axs[0].set_yticklabels([f"{y:.1f}" for y in [0.2, 0.4, 0.6]], fontproperties=prop4, fontsize=22, rotation=90, color='black')
axs[0].tick_params(axis='both', labelsize=24, colors='black')

# Plot QP 35
plot_pdf(axs[1], qp35, label="QP 35", color=c2)
axs[1].set_xlabel("Relative SSIM (%)", fontproperties=prop4, fontsize=24, color='black')
axs[1].legend(fontsize=26, loc='upper left', frameon=True)
axs[1].set_xlim(0.85, 1.0)
axs[1].set_ylim(0, 0.6)
axs[1].grid(True, linestyle='--', linewidth=0.5, color='lightgray', alpha=0.9)
# Set spine colors for second subplot
axs[1].spines['top'].set_color('lightgray')
axs[1].spines['right'].set_color('lightgray')
axs[1].spines['bottom'].set_color('black')
axs[1].spines['left'].set_color('black')
# Set spine widths
axs[1].spines['bottom'].set_linewidth(1.5)
axs[1].spines['left'].set_linewidth(1.5)
axs[1].set_xticks([0.85, 0.9, 0.95, 1.0])
axs[1].set_xticklabels([f"{x*100}%" for x in [0.85, 0.9, 0.95, 1.0]], fontproperties=prop4, fontsize=16, color='black')
axs[1].set_yticks([0.2, 0.4, 0.6])
axs[1].set_yticklabels([f"{y:.1f}" for y in [0.2, 0.4, 0.6]], fontproperties=prop4, fontsize=16, rotation=90, color='black')
axs[1].tick_params(axis='both', labelsize=24, colors='black')

# Shared Y label
fig.text(0.04, 0.5, "PDF", va='center', rotation='vertical',
         fontproperties=prop4, fontsize=24, color='black')

plt.tight_layout(rect=[0.08, 0.03, 1, 0.98])  # Adjust layout to make space for y-label

# plt.tight_layout()
plt.savefig("graph/cumulative_ssim_distribution_split.pdf", dpi=2400)