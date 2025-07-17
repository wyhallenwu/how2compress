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

plt.rcParams["font.family"] = [prop1.get_name(), prop2.get_name()]
plt.rcParams["font.weight"] = "bold"

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
def plot_pdf(data_dict, label, color, linestyle='-'):
    x = np.array(sorted(data_dict.keys()))
    y = np.array([data_dict[k] for k in x])
    
    x_smooth = np.linspace(0, 1, 300)
    spline = make_interp_spline(x, y, k=3)
    y_smooth = spline(x_smooth)
    
    plt.plot(x_smooth, y_smooth, label=label, color=color, linestyle=linestyle, linewidth=3)

# Start figure
plt.figure(figsize=(6, 6))

# Plot PDFs
plot_pdf(qp45, label="QP 45", color=c1)
plot_pdf(qp35, label="QP 35", color=c2)

# === Add vertical lines ===
for x, y in qp45_raw.items():
    plt.vlines(x, 0, y, color=c1, linestyle='--', linewidth=1.5)

for x, y in qp35_raw.items():
    plt.vlines(x, 0, y, color=c2, linestyle='--', linewidth=1.5)


# Labels, grid, and legend
plt.xlabel("Relative SSIM", fontproperties=prop1, fontsize=18)
plt.ylabel("Probability Density", fontproperties=prop1, fontsize=18)
plt.title("", fontproperties=prop1, fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend(fontsize=18, loc='upper left', frameon=True)
plt.xlim(0.85, 1.0)
plt.ylim(0, 0.6)
xticks = [0.85, 0.9, 0.95, 1.0]
xtick_labels = [f"{x:.2f}x" for x in xticks]
plt.xticks(xticks, xtick_labels, fontproperties=prop1, fontsize=16)
yticks = [0.2, 0.4, 0.6]
ytick_labels = [f"{y:.1f}" for y in yticks]
plt.yticks(yticks, ytick_labels, fontproperties=prop1, fontsize=16, rotation=90)
plt.tight_layout()


plt.savefig("graph/cumulative_ssim_distribution.pdf", dpi=2400)