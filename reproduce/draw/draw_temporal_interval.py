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
verdanabi= fm.FontProperties(fname=font1_path)

plt.rcParams["font.family"] = prop.get_name()
plt.rcParams["font.weight"] = "bold"

# Define colors
red = "#D77071"  
blue = "#6888F5"  

# Data
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

# Convert to DataFrame
df_mot = pd.DataFrame({"Interval": list(interval_performance_on_mot.keys()),
                        "Accuracy": list(interval_performance_on_mot.values()),
                        "Category": "MOT"})

df_aicity = pd.DataFrame({"Interval": list(interval_performance_on_aicity.keys()),
                           "Accuracy": list(interval_performance_on_aicity.values()),
                           "Category": "AICITY"})

# Combine DataFrames
df = pd.concat([df_mot, df_aicity])

# Set color palette
palette = {"MOT": red, "AICITY": blue}

# Plot
plt.figure(figsize=(5, 2.3))
sns.lineplot(data=df, x="Interval", y="Accuracy", hue="Category", marker="o", linewidth=2.5, palette=palette, alpha=0.85)

# Formatting
plt.xlabel("Interval", fontsize=12, fontweight="bold")
plt.ylabel("Accuracy", fontsize=12, fontweight="bold")
plt.xticks(fontsize=10, fontweight="bold")
plt.yticks([0.8, 1], labels=["0.8x", "1x"], fontsize=10, fontweight="bold")  # Set y-axis ticks
plt.ylim(0.6, 1.1)  # Ensure y-axis is within range
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="Category", fontsize=12)
plt.tight_layout()
plt.savefig("graph/interval.pdf", dpi=2400)