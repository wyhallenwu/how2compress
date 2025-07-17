import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
from itertools import cycle

# Load the correct font
font_path = "/usr/share/fonts/truetype/msttcorefonts/arialbd.ttf"
prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
font1_path = "/usr/share/fonts/truetype/msttcorefonts/Verdana_Bold_Italic.ttf"
fm.fontManager.addfont(font1_path)
verdanabi= fm.FontProperties(fname=font1_path)

plt.rcParams["font.family"] = prop.get_name()
plt.rcParams["font.weight"] = "bold"

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
fig, axes = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [3, 2, 3]})

# First subplot: Line plot (Compression Rate vs Backbone)
sns.lineplot(data=df1, x="Backbone", y="Compression Rate", marker="o", ax=axes[0], color=blue, linewidth=8, markersize=12, alpha=0.85)
axes[0].set_ylabel("Compression Rate", fontsize=24)
axes[0].set_xlabel("Backbone", fontsize=24)
axes[0].grid(True, linestyle="--")
axes[0].tick_params(axis="both", labelsize=24)
axes[0].tick_params(axis="x", labelsize=24, rotation=0)

# Second subplot: Bar plot (Accuracy Variance vs Detector)
barplot1 = sns.barplot(data=df2, x="Detector", y="Accuracy Deviation", hue="Method", ax=axes[1], palette=[red, blue], alpha=0.85)
axes[1].set_ylabel("Accuracy Deviation", fontsize=24)
axes[1].set_xlabel("Detector", fontsize=24)
axes[1].legend(title="", fontsize=18)
axes[1].tick_params(axis="both", labelsize=24)
axes[1].set_yticklabels([]) 
axes[1].tick_params(axis="y", labelsize=24, rotation=45)  
axes[1].grid(True, linestyle="--")

# Third subplot: Line plot (Interval vs Accuracy)
sns.lineplot(ax=axes[2], data=df, x="Interval", y="Accuracy", hue="Category", marker="o", linewidth=8, palette=palette, alpha=0.85)

axes[2].set_xlabel("Interval", fontsize=24, fontweight="bold")
axes[2].set_ylabel("Accuracy", fontsize=24, fontweight="bold")
axes[2].set_xticks(df["Interval"].unique())
axes[2].set_yticks([0.8, 1], labels=["0.8x", "1x"])
axes[2].set_ylim(0.6, 1.1)
axes[2].grid(True, linestyle="--", alpha=0.6)
axes[2].legend(title="", fontsize=18)
axes[2].tick_params(axis="y", labelsize=24, rotation=90)  
axes[2].tick_params(axis="both", labelsize=24)

# Apply hatching to barplot
hatch_styles = cycle(["\\", "/"])

for bar in barplot1.patches:
    bar.set_hatch(next(hatch_styles))
    bar.set_edgecolor("white")  # Ensure hatching is visible

# Adjust ticks for all subplots
axes[0].set_yticks([0.49, 0.50, 0.51])
axes[1].set_yticks([-0.1, -0.025, 0, 0.025, 0.05])

# Add value labels for Compression Rate (First subplot)
for i, (x, y) in enumerate(zip(df1["Backbone"], df1["Compression Rate"])):
    axes[0].text(i, y+0.002, f"{y:.3f}", ha="center", fontsize=24, fontweight="bold", color="black")

# Add value labels for Accuracy Variance (Second subplot)
for bar in axes[1].containers:
    axes[1].bar_label(bar, fmt="%.3f", padding=3, fontsize=24, fontweight="bold", color="black")

# axes[1].margins(0)
axes[2].margins(0)
plt.subplots_adjust(wspace=0)
# Adjust layout and display the plots
plt.tight_layout()

# Save figure
plt.savefig("graph/ablation.pdf", dpi=2400)