import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
from itertools import cycle

# Load the correct font
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

# Data for first two subplots
result1 = {
    "segformer": 0.499,
    "mobilevit-v2": 0.504,
    "deeplabv3-vit": 0.497,
}

result2 = {
    "yolov5x": [-0.045, -0.032],  # ['accmpeg', 'ours']
    "detr": [-0.018, 0.021],  # ['accmpeg', 'ours']
}

red = "#D77071"
blue = "#6888F5"

# Convert to DataFrames
df1 = pd.DataFrame(result1.items(), columns=["Backbone", "Compression Rate"])
df2 = pd.DataFrame(
    [(detector, method, value) for detector, values in result2.items() for method, value in zip(["AccMPEG", "Ours"], values)],
    columns=["Detector", "Method", "Accuracy Deviation"],
)

# Data for third subplot (Interval vs. Accuracy)
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
palette = {"MOT": "#8E7FB8", "AICITY": "#A2C9AE"}

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={"width_ratios": [3, 3, 3]})

# Define markers for each backbone
marker_map = {
    "segformer": "o",        # Circle
    "mobilevit-v2": "s",     # Square
    "deeplabv3-vit": "D",    # Diamond
}
colors_map = {
    "segformer": "#0E986F",       # reddish
    "mobilevit-v2": "#796CAD",    # teal green
    "deeplabv3-vit": "#D65813",   # dark blue
}

# Plot each backbone individually to customize markers
for i, row in df1.iterrows():
    axes[0].plot(
        row["Backbone"],
        row["Compression Rate"],
        marker=marker_map[row["Backbone"]],
        markersize=18,
        color=colors_map[row["Backbone"]],
        linewidth=0,
        label=row["Backbone"]
    )

# Connect them with a line
axes[0].plot(df1["Backbone"], df1["Compression Rate"],
             color=blue, linewidth=4, alpha=0.85, zorder=0)

# Axis labels and formatting
axes[0].set_ylabel("Compression Rate", fontsize=26, labelpad=20, fontproperties=prop6)
axes[0].set_xlabel("Backbone", fontsize=26, labelpad=30, fontproperties=prop6)
axes[0].grid(True, linestyle="--")
axes[0].tick_params(axis="y", labelsize=24, rotation=0, left=False, labelleft=False)
axes[0].tick_params(axis="x", labelsize=24, rotation=0, bottom=False, labelbottom=False)

# Custom legend
legend1 = axes[0].legend(title="", fontsize=24, loc="upper right", frameon=True, prop=prop5)
# Explicitly set legend font size to 24
for text in legend1.get_texts():
    text.set_fontproperties(prop6)
    text.set_fontsize(24)

axes[0].set_yticks([0.495, 0.515, 0.535])

# Second subplot: Bar plot (Accuracy Variance vs Detector)
barplot1 = sns.barplot(data=df2, x="Detector", y="Accuracy Deviation", hue="Method", ax=axes[1], palette=[red, blue], alpha=0.85)
axes[1].set_ylabel("Accuracy Deviation", fontsize=26, fontproperties=prop6) 
axes[1].set_xlabel("Detector", fontsize=26, fontproperties=prop6)
legend2 = axes[1].legend(title="", fontsize=24, prop=prop6)
# Explicitly set legend font size to 24
for text in legend2.get_texts():
    text.set_fontproperties(prop6)
    text.set_fontsize(24)

axes[1].tick_params(axis="both", labelsize=24)
axes[1].set_yticklabels([]) 
axes[1].tick_params(axis="y", labelsize=24, rotation=45)  
axes[1].grid(True, linestyle="--")

# Third subplot: Line plot (Interval vs Accuracy)
sns.lineplot(ax=axes[2], data=df, x="Interval", y="Accuracy", hue="Category", marker="o", linewidth=8, palette=palette, alpha=0.85)

axes[2].set_xlabel("Interval", fontsize=26, fontproperties=prop6)
axes[2].set_ylabel("Accuracy", fontsize=26, fontproperties=prop6)
axes[2].set_xticks(df["Interval"].unique())
axes[2].set_yticks([0.8, 1], labels=["0.8x", "1x"])
axes[2].set_ylim(0.6, 1.1)
axes[2].grid(True, linestyle="--", alpha=0.6)
legend3 = axes[2].legend(title="", fontsize=24, prop=prop6)
# Explicitly set legend font size to 24
for text in legend3.get_texts():
    text.set_fontproperties(prop6)
    text.set_fontsize(24)

axes[2].tick_params(axis="y", labelsize=24, rotation=90)  
axes[2].tick_params(axis="both", labelsize=24)

# Apply font properties to all tick labels
for ax in axes:
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(prop5)
        tick.set_fontsize(24)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(prop5)
        tick.set_fontsize(24)

# Apply hatching to barplot
hatch_styles = cycle(["\\", "/"])

for bar in barplot1.patches:
    bar.set_hatch(next(hatch_styles))
    bar.set_edgecolor("white")  # Ensure hatching is visible

# Adjust ticks for all subplots
axes[1].set_yticks([-0.1, -0.025, 0, 0.025, 0.05])

# Add value labels for Compression Rate (First subplot)
for i, (x, y) in enumerate(zip(df1["Backbone"], df1["Compression Rate"])):
    axes[0].text(i, y+0.002, f"{y:.3f}", ha="center", fontsize=24, color="black", fontproperties=prop6)

# Add value labels for Accuracy Variance (Second subplot)
for bar in axes[1].containers:
    labels = axes[1].bar_label(bar, fmt="%.3f", padding=3, fontsize=24,color="black")
    for label in labels:
        label.set_fontproperties(prop6)
        label.set_fontsize(24)

axes[2].margins(0)

plt.subplots_adjust(wspace=0)
# Adjust layout and display the plots
plt.tight_layout()

# Save figure
plt.savefig("graph/ablation.pdf", dpi=2400)