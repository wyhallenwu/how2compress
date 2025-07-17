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
# Colors
c1 = "#91CAE8"
c2 = "#F48892"
red = "#D77071"
blue = "#6888F5"

# Set up GridSpec for 2 rows, 2 columns
fig = plt.figure(figsize=(16, 7))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.1], height_ratios=[1, 1], wspace=0.2, hspace=0.2)

# Left column: two stacked subplots
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
# Right column: bar plot spanning both rows
ax3 = fig.add_subplot(gs[:, 1])

# Original data for PDF plots
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

# Top PDF (QP 45)
plot_pdf(ax1, qp45, label="QP 45", color=c1)
ax1.legend(fontsize=30, loc='upper left', frameon=True)
ax1.set_ylim(0, 0.6)
ax1.set_xlim(0.85, 1.0)
ax1.grid(True, linestyle='--', linewidth=0.5, color='lightgray', alpha=0.9)
ax1.spines['top'].set_color('lightgray')
ax1.spines['right'].set_color('lightgray')
ax1.spines['bottom'].set_color('black')
ax1.spines['left'].set_color('black')
ax1.spines['bottom'].set_linewidth(2)
ax1.spines['left'].set_linewidth(2)
ax1.set_xticks([0.85, 0.9, 0.95, 1.0])
ax1.set_xticklabels(['', '', '', ''])  # Remove x tick labels on top
ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Remove x ticks on top
ax1.set_yticks([0.2, 0.4, 0.6])
ax1.set_yticklabels([f"{y:.1f}" for y in [0.2,  0.4, 0.6]], fontproperties=prop5, fontsize=30, rotation=90, color='black')
ax1.tick_params(axis='both', labelsize=30, colors='black')
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x*100)}%" if 0.85 <= x <= 1.0 else ""))

# Bottom PDF (QP 35)
plot_pdf(ax2, qp35, label="QP 35", color=c2)
ax2.legend(fontsize=30, loc='upper left', frameon=True)
ax2.set_ylim(0, 0.6)
ax2.set_xlim(0.85, 1.0)
ax2.grid(True, linestyle='--', linewidth=0.5, color='lightgray', alpha=0.9)
ax2.spines['top'].set_color('lightgray')
ax2.spines['right'].set_color('lightgray')
ax2.spines['bottom'].set_color('black')
ax2.spines['left'].set_color('black')
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
ax2.set_xlabel("SSIM Preservation", fontproperties=prop6, fontsize=34, color='black')
ax2.set_xticks([0.85, 0.9, 0.95, 1.0])
ax2.set_xticklabels([f"{x*100:.0f}%" for x in [0.85, 0.9, 0.95, 1.0]], fontproperties=prop5, fontsize=30, color='black')
ax2.set_yticks([0.2, 0.4, 0.6])
ax2.set_yticklabels([f"{y:.1f}" for y in [0.2, 0.4, 0.6]], fontproperties=prop5, fontsize=30, rotation=90, color='black')
ax2.tick_params(axis='both', labelsize=30, colors='black')
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x*100)}%" if 0.85 <= x <= 1.0 else ""))

# Shared Y label for PDFs (move further left)
fig.text(0.01, 0.5, "PDF", va='center', rotation='vertical', fontproperties=prop6, fontsize=34, color='black')

# Prepare data for bar plot
data = {
    "Mask": [30, 35, 40, 45, 50] * 2,
    "Accuracy": [1.00, 0.96, 0.92, 0.85, 0.75,
                 1.00, 0.96, 0.94, 0.92, 0.89],
    "Type": ["RoI"] * 5 + ["BG"] * 5
}
df = pd.DataFrame(data)

# Create bar plot on the right subplot
sns.barplot(
    data=df,
    y="Mask", x="Accuracy", hue="Type",
    palette={"RoI": red, "BG": blue},
    edgecolor="black", alpha=0.85,
    orient="h",
    ax=ax3
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
ax3.barh(y - bar_height/2, roi_acc, height=bar_height, facecolor="none", edgecolor="white",
        linewidth=1.5, hatch='\\')
ax3.barh(y + bar_height/2, bg_acc, height=bar_height, facecolor="none", edgecolor="white",
        linewidth=1.5, hatch='/')

# Label values on bars
for i, bar in enumerate(ax3.patches):
    width = bar.get_width()
    y_pos = bar.get_y() + bar.get_height() / 2
    if i not in [0, 1, 5, 12, 17, 6, 13, 18]:
        ax3.text(width + 0.002, y_pos, f"{width:.2f}×",
                va='center', ha='left', fontsize=30, fontproperties=prop5)
    elif i in [0, 1]:
        ax3.text(width + 0.002, y_pos, f"{width:.2f}×",
                va='center', ha='left', fontsize=30, fontproperties=prop5)

# Axis formatting for bar plot
ax3.set_xlim(0.7, 1.05)
ax3.set_xlabel("Accuracy Degradation", fontproperties=prop6, fontsize=34)
ax3.set_ylabel("QP", fontproperties=prop6, fontsize=34)
ax3.set_yticks(y)
ax3.set_yticklabels([str(v) for v in y_labels], fontproperties=prop5, fontsize=30, rotation=90)
ax3.set_xticks([0.7, 0.8, 0.9, 1.0])
ax3.set_xticklabels([f"{x:.2f}×" for x in [0.7, 0.8, 0.9, 1.0]], fontproperties=prop5, fontsize=30)
ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}×" if 0.7 <= x <= 1.0 else ""))
ax3.grid(axis="x", linestyle="--", linewidth=0.5, color='lightgray', alpha=0.9)

# Set spine colors for bar plot
ax3.spines['top'].set_color('lightgray')
ax3.spines['right'].set_color('lightgray')
ax3.spines['bottom'].set_color('black')
ax3.spines['left'].set_color('black')
# Set spine widths for bar plot
ax3.spines['bottom'].set_linewidth(2)
ax3.spines['left'].set_linewidth(2)

# Custom legend for bar plot with correct font size and properties
legend_font = fm.FontProperties(fname=font_path6, size=28)
bar_legend_patches = [
    mpatches.Patch(facecolor=red, edgecolor="white", label="RoI", hatch='\\'),
    mpatches.Patch(facecolor=blue, edgecolor="white", label="BG", hatch='/')
]
ax3.legend(handles=bar_legend_patches, title="", loc="lower right", frameon=True, prop=legend_font)

ax3.tick_params(axis='both', labelsize=30, colors='black')

plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.18, wspace=0.2, hspace=0.2)

plt.savefig("graph/design_motivation.pdf", dpi=2400) 