import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"
# plt.rcParams["font.weight"] = "bold"
# Data preparation
data = {
    "Resolution": ["1080p", "900p", "720p", "480p"],
    "FLOPS": [7.256, 3.023, 1.934, 0.8688],  # FLOPS in GFLOPS
    "MACs": [
        3.582,
        1.493,
        0.9556,
        0.4292,
    ],  # MACs in GMACs/MMACs converted to GMACs for consistency
}

# Create a DataFrame
df = pd.DataFrame(data)

# Custom colors and markers
colors = ["#6A8EC9", "#E84446", "#59B78F", "#7A378A"]
markers = ["D", "o", "s", "p"]

# Set up the matplotlib figure
plt.figure(figsize=(4, 3.3))

# Plot using seaborn
sns.scatterplot(
    data=df,
    x="MACs",
    y="FLOPS",
    hue="Resolution",
    s=100,  # Size of the markers
    style="Resolution",  # Differentiating markers by resolution
    edgecolor="black",  # Add black edges to the markers
    linewidth=2,  # Width of the marker edges
    palette=colors,  # Custom colors
    markers=markers,  # Custom markers
)

# Customizing the title and labels
plt.title("", fontsize=10, fontweight="bold")
plt.xlabel(
    "GMACs",
    fontsize=16,
    # fontweight="bold",
)
plt.ylabel(
    "GFLOPS",
    fontsize=16,
    # fontweight="bold",
)

# Adjust tick parameters for font size
plt.xticks([0.5, 1.5, 2.5, 3.5], fontsize=14, fontweight="bold")
plt.yticks([1, 3, 6], fontsize=14, fontweight="bold")
plt.grid(True, color="grey", linewidth=0.3, linestyle="--")

# Show the legend with bold text
plt.legend(
    title="",
    title_fontsize=16,
    fontsize=16,
    ncol=2,
    prop={"weight": "bold"},
)
plt.tight_layout()
plt.savefig("graph/method2.pdf", dpi=1200, bbox_inches="tight")
