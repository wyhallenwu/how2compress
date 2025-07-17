import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


plt.rcParams["font.family"] = "Arial"
# plt.rcParams["font.weight"] = "bold"

# Data for devices
devices = [
    "Orin Nano (4GB)",
    "Orin Nano (8GB)",
    "Xavier NX",
    "RTX 3090",
]

flops_fp32 = np.array([640, 1280, 844.8, 35.58 * 1000])  # FLOPS in GFLOPS

# Data for the 1080p model
flops_1080p = 7.256  # FLOPS in GFLOPS
# Colors for each bar
colors = ["#6A8EC9", "#E84446", "#59B78F", "#7A378A"]
# Create the bar plot
plt.figure(figsize=(5.5, 3.3))
ax = sns.barplot(
    x=devices,
    y=np.log(flops_fp32),
    hue=devices,
    palette=colors,
    linewidth=2,
    edgecolor="black",
)

# Annotate each bar with the true FLOPS value
for i, value in enumerate(flops_fp32):
    ax.text(
        i,
        np.log(value) + 0.1,
        f"{value}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

# Add a line for the 1080p model FLOPS
baseline_line = plt.axhline(
    np.log(flops_1080p),
    color="red",
    linestyle="--",
    label="Model FLOPS Requirement: 7.2 GFLOPS",
    linewidth=2,
)

# Add a legend for the dashed line
plt.legend(
    handles=[baseline_line],
    loc="upper left",
    fontsize=14,
    prop={"weight": "bold"},
)

# Turn off y-axis ticks and labels
ax.tick_params(axis="y", which="both", left=False, labelleft=False)
# ax.tick_params(axis="x", labelsize=7)
ax.set_ylim(0, 13)

# Set labels and title
ax.set_xlabel("Devices", fontsize=16)
ax.set_ylabel(
    "Device Max GFLOPS (fp32)",
    fontsize=15,
    # fontweight="bold",
)
# Adjust tick parameters for font size
plt.xticks(
    fontsize=11,
    # fontweight="bold",
)
plt.yticks(
    fontsize=8,
    fontweight="bold",
)
plt.tight_layout()

# Save the figure as a PDF
plt.savefig("graph/method1.pdf", dpi=1200, bbox_inches="tight")
