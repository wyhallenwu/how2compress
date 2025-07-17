import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.font_manager as fm
from matplotlib.ticker import LogLocator, ScalarFormatter

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

# Data for the first plot
data1 = {
    "Dataset": ["Scene A", "Scene B", "Scene C", "Scene D"],
    "AS": [8918, 3968, 8960, 10875],
    "FL": [4075, 1646, 3985, 5026],
    "AQ": [2914, 1361, 3258, 3877],
}
df1 = pd.DataFrame(data1)
df1["Frame-level"] = (1 - df1["FL"] / df1["AS"]) * 100
df1["Macroblock-level"] = (1 - df1["AQ"] / df1["AS"]) * 100
df1["Improvement"] = df1["Frame-level"] - df1["Macroblock-level"]
df1_melted = df1.melt(
    id_vars="Dataset",
    value_vars=["Frame-level", "Macroblock-level"],
    var_name="Metric",
    value_name="Percentage",
)

# Data for the second plot
data2 = {
    "Dataset": ["Scene B", "Scene D"],
    "AS": [3968, 10805],
    "CG": [1436, 4517],
    "AQ": [1361, 3877],
}
df2 = pd.DataFrame(data2)
df2["Fine-grained"] = (1 - df2["AQ"] / df2["AS"]) * 100
df2["Coarse-grained"] = (1 - df2["CG"] / df2["AS"]) * 100
df2["Improvement"] = df2["Fine-grained"] - df2["Coarse-grained"]
df2_melted = df2.melt(
    id_vars="Dataset",
    value_vars=["Fine-grained", "Coarse-grained"],
    var_name="Metric",
    value_name="Percentage",
)

# plt.rcParams["font.family"] = "Arial"
# plt.rcParams["font.weight"] = "bold"
# Create subplots
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [0.6, 0.4]}
)

# First subplot
sns.barplot(
    x="Dataset",
    y="Percentage",
    hue="Metric",
    data=df1_melted,
    errorbar=None,
    palette=["#72bb6c", "#fcaa57"],
    edgecolor="black",
    linewidth=1,
    width=0.8,
    ax=ax1,
)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
for i, dataset in enumerate(df1["Dataset"]):
    macroblock_y = df1.loc[i, "Macroblock-level"]
    frame_y = df1.loc[i, "Frame-level"]
    improvement = df1.loc[i, "Improvement"]

    ax1.plot(
        [i - 0.2, i + 0.2],
        [macroblock_y, macroblock_y],
        color="black",
        linestyle="--", linewidth=4
    )
    ax1.plot([i - 0.2, i + 0.2], [frame_y, frame_y], color="black", linestyle="--", linewidth=4)
    ax1.annotate(
        "",
        xy=(i - 0.1, frame_y),
        xytext=(i - 0.1, macroblock_y),
        arrowprops=dict(
            facecolor="black", shrink=0.1, width=2, headwidth=15, headlength=20
        ),
    )
    ax1.text(
        i - 0.25,
        (frame_y + macroblock_y) / 2,
        f"{abs(improvement):.1f}%",
        ha="center",
        va="center",
        fontsize=30,
        color="black",
        rotation=90,
        zorder=10,
        # fontweight="bold",
    )

ax1.set_xlabel("", fontsize=30, fontproperties=prop5)
ax1.set_ylabel("Bitrate Saving (Raw QP 25)", fontproperties=prop5, fontsize=34)
ax1.set_xticks(ax1.get_xticks())
ax1.set_xticklabels(ax1.get_xticklabels(), fontproperties=prop5,fontsize=36)
ax1.set_yticks([55, 60, 65, 70])
ax1.set_yticklabels([f"{int(y)}%" for y in ax1.get_yticks()], fontproperties=prop5, fontsize=34, rotation=90)
# print(f"ax1.get_xticks(): {ax1.get_xticks()}")
# print(f"ax1.get_yticks(): {ax1.get_yticks()}")

ax1.set_ylim(50, 70)

# Apply hatches to the bars
hatches = ["x", "-"]  # Hatch patterns for BG and RoI
bars = ax1.patches  # Get all the bars

for i, bar in enumerate(bars):
    # Alternate between the hatches based on the hue (Category)
    # bar.set_hatch(hatches[0 if i < (len(bars) // 2 - 1) else 1])  # Apply hatch
    bar.set_hatch(hatches[0 if i < (len(bars) // 2 - 1) else 1])
    bar.set_edgecolor("white")
    bar.set_linewidth(0.1)

fl_patch = Rectangle(
    (0, 0), 1, 1, facecolor="#72bb6c", edgecolor="white", linewidth=10, hatch="x", hatch_linewidth=4
)
mb_patch = Rectangle(
    (0, 0), 1, 1, facecolor="#fcaa57", edgecolor="white", linewidth=10, hatch="-", hatch_linewidth=4
)
legend_font = fm.FontProperties(fname=font_path6, size=22)

ax1.legend(
    [fl_patch, mb_patch],
    ["Frame-Level(when2compress)", "Macroblock-Level(Codec's AQ)"],
    title="",
    fontsize=22,
    title_fontsize=22,
    handlelength=1.5,
    handleheight=1,
    loc="upper right",
    prop=legend_font,
)
ax1.get_legend().set_zorder(0)

# Second subplot
sns.barplot(
    x="Dataset",
    y="Percentage",
    hue="Metric",
    data=df2_melted,
    errorbar=None,
    palette=["#F38276", "#91CAE8"],
    hue_order=["Coarse-grained", "Fine-grained"],
    edgecolor="black",
    linewidth=1,
    width=0.8,
    ax=ax2,
)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
for i, dataset in enumerate(df2["Dataset"]):
    macroblock_y = df2.loc[i, "Fine-grained"]
    frame_y = df2.loc[i, "Coarse-grained"]
    improvement = df2.loc[i, "Improvement"]

    ax2.plot(
        [i - 0.2, i + 0.2],
        [macroblock_y, macroblock_y],
        color="black",
        linestyle="--", linewidth=4
    )
    ax2.plot([i - 0.2, i + 0.2], [frame_y, frame_y], color="black", linestyle="--", linewidth=4)
    ax2.annotate(
        "",
        xy=(i - 0.1, frame_y),
        xytext=(i - 0.1, macroblock_y),
        arrowprops=dict(
            facecolor="black", shrink=0.1, width=2, headwidth=15, headlength=20
        ),
    )
    ax2.text(
        i - 0.25,
        (frame_y + macroblock_y) / 2,
        f"{abs(improvement):.1f}%",
        ha="center",
        va="center",
        fontsize=30,
        color="black",
        rotation=90,
        zorder=10,
        # fontweight="bold",
    )

ax2.set_xlabel("", fontsize=30, fontproperties=prop5)
ax2.set_ylabel("", fontsize=30, fontproperties=prop5)
ax2.set_xticks(ax2.get_xticks())
ax2.set_xticklabels(ax2.get_xticklabels(), fontproperties=prop5, fontsize=36)
# ax2.set_yticks(ax2.get_yticks())
ax2.set_yticks([])
# ax2.set_yticklabels([f"{int(y)}%" for y in ax2.get_yticks()], fontsize=6)

ax2.set_ylim(50, 70)

# Apply hatches to the bars
hatches = ["+", "*"]  # Hatch patterns for BG and RoI
bars = ax2.patches  # Get all the bars

for i, bar in enumerate(bars):
    # Alternate between the hatches based on the hue (Category)
    # bar.set_hatch(hatches[0 if i < (len(bars) // 2 - 1) else 1])  # Apply hatch
    bar.set_hatch(hatches[0 if i < (len(bars) // 2 - 1) else 1])
    bar.set_edgecolor("white")
    bar.set_linewidth(0.1)
cg_patch = Rectangle(
    (0, 0), 1, 1, facecolor="#F38276", edgecolor="white", linewidth=10, hatch="+", hatch_linewidth=4
)
finel_patch = Rectangle(
    (0, 0), 1, 1, facecolor="#91CAE8", edgecolor="white", linewidth=10, hatch="*", hatch_linewidth=4
)
legend_font = fm.FontProperties(fname=font_path6, size=22)
ax2.legend(
    [cg_patch, finel_patch],
    ["Coarse(where2compress)", "Fine(Codec's AQ)"],
    title="",
    fontsize=24,
    title_fontsize=30,
    handlelength=1.5,
    handleheight=1,
    loc="upper right",
    prop=legend_font,
)
ax2.get_legend().set_zorder(0)

# Adjust spacing between subplots
plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.18, wspace=0.2)

# Save combined figure
plt.savefig(
    "graph/motivation.pdf", dpi=1200, bbox_inches="tight", pad_inches=0.1
)
# plt.show()
