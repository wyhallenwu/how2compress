import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import gridspec

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.weight"] = "bold"
# Data from the table
data = {
    "Resolution": [
        "1080p",
        "900p",
        "720p",
        "480p",
        "1080p",
        "900p",
        "720p",
        "480p",
        "1080p",
        "900p",
        "720p",
        "480p",
        "1080p",
        "900p",
        "720p",
        "480p",
    ],
    "Inference Time (ms)": [
        10.8,
        9.8,
        9.5,
        9.4,
        32,
        31,
        30.3,
        29.5,
        12.3,
        10.4,
        9.5,
        9.2,
        32.7,
        31.6,
        31.1,
        25.2,
    ],
    "Peak Memory Usage (MB)": [
        62,
        48,
        37,
        24,
        166,
        151,
        153,
        150,
        118,
        84,
        58.2,
        25,
        234,
        203,
        180,
        151,
    ],
    "Device": [
        "RTX3090",
        "RTX3090",
        "RTX3090",
        "RTX3090",
        "Jetson Orin Nano",
        "Jetson Orin Nano",
        "Jetson Orin Nano",
        "Jetson Orin Nano",
        "RTX3090",
        "RTX3090",
        "RTX3090",
        "RTX3090",
        "Jetson Orin Nano",
        "Jetson Orin Nano",
        "Jetson Orin Nano",
        "Jetson Orin Nano",
    ],
    "Method": [
        "Tetris",
        "Tetris",
        "Tetris",
        "Tetris",
        "Tetris",
        "Tetris",
        "Tetris",
        "Tetris",
        "AccMPEG",
        "AccMPEG",
        "AccMPEG",
        "AccMPEG",
        "AccMPEG",
        "AccMPEG",
        "AccMPEG",
        "AccMPEG",
    ],
}

colors = {"Tetris": "#6A8EC9", "AccMPEG": "#E84446"}
# colors = ["#6A8EC9", "#E84446", "#59B78F", "#7A378A"]
markers = ["D", "o", "s", "p"]
# Creating a DataFrame
df = pd.DataFrame(data)

markers = {"1080p": "D", "900p": "o", "720p": "s", "480p": "p"}


def draw1():
    # Scatter plot for RTX3090
    figure = plt.figure(figsize=(7, 2.9))
    gs = gridspec.GridSpec(1, 3, width_ratios=[5, 5, 1.1])
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[0, 2])
    sns.lineplot(
        data=df[df["Device"] == "Jetson Orin Nano"],
        x="Inference Time (ms)",
        y="Peak Memory Usage (MB)",
        hue="Method",
        style="Resolution",
        edgecolor="black",
        linewidth=2,
        s=75,
        palette=colors,
        markers=markers,
        legend=False,
        ax=ax1,
    )
    ax1.set_xlabel(
        "Inference Time (ms)",
    )
    ax1.set_ylabel(
        "Peak GPU Memory Usage (MB)",
    )
    ax1.set_xticks([20, 25, 30, 35, 40])
    ax1.set_yticks([130, 150, 175, 200, 225, 250])
    ax1.set_xlabel("Emphasis Prediction Time (ms)", fontsize=12)
    ax1.set_ylabel("Peak GPU Memory Usage (MB)", fontsize=12)
    ax1.grid(True, color="grey", linewidth=0.5, linestyle="--")

    sns.scatterplot(
        data=df[df["Device"] == "RTX3090"],
        x="Inference Time (ms)",
        y="Peak Memory Usage (MB)",
        hue="Method",
        style="Resolution",
        # edgecolor="black",
        linewidth=2,
        s=75,
        palette=colors,
        markers=markers,
        legend=False,
        ax=ax2,
    )
    ax2.set_xlabel(
        "Inference Time (ms)",
    )
    ax2.set_ylabel(
        "Peak GPU Memory Usage (MB)",
    )
    ax2.set_xticks([9, 10, 11, 12, 13])
    ax2.set_yticks([25, 50, 75, 100, 125, 150])
    ax2.set_xlabel("Emphasis Prediction Time (ms)", fontsize=12)
    ax2.set_ylabel("Peak GPU Memory Usage (MB)", fontsize=12)
    ax2.grid(True, color="grey", linewidth=0.5, linestyle="--")

    method_elements = []
    resolution_elements = []

    # Method legend (colors)
    for method, color in colors.items():
        method_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=method,
                markerfacecolor=color,
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=2,
            )
        )

    # Resolution legend (markers)
    for resolution, marker in markers.items():
        resolution_elements.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                label=resolution,
                markerfacecolor="gray",
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=2,
            )
        )

    ax3.axis("off")

    # Create two separate legends
    method_legend = ax3.legend(
        handles=method_elements,
        title="Method",
        loc="upper left",
        bbox_to_anchor=(-2, 1),
        fontsize=10,
        prop={"weight": "bold"},
        handletextpad=0.5,
    )

    # Add the method legend to the plot
    ax3.add_artist(method_legend)

    # Create and add the resolution legend
    resolution_legend = ax3.legend(
        handles=resolution_elements,
        title="Resolution",
        loc="upper left",
        bbox_to_anchor=(-2, 0.5),
        fontsize=10,
        prop={"weight": "bold"},
        handletextpad=0.5,
    )

    plt.grid(True, color="grey", linewidth=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig("graph/model-consumption-1.pdf", dpi=1200, bbox_inches="tight")


def draw2():
    # Scatter plot for Jetson Orin Nano
    plt.figure(figsize=(3.2, 2.9))
    sns.scatterplot(
        data=df[df["Device"] == "Jetson Orin Nano"],
        x="Inference Time (ms)",
        y="Peak Memory Usage (MB)",
        hue="Resolution",
        s=100,
        style="Resolution",
        edgecolor="black",
        linewidth=2,
        palette=colors,
        markers=markers,
    )
    plt.xlabel(
        "Inference Time (ms)",
        fontsize=12,
        # fontweight="bold",
    )
    plt.ylabel(
        "Peak GPU Memory Usage (MB)",
        fontsize=12,
        # fontweight="bold",
    )
    plt.legend(
        title="",
        title_fontsize=12,
        fontsize=12,
        ncol=2,
        prop={"weight": "bold"},
    )
    plt.xticks(
        [29, 30, 31, 32],
        fontsize=10,
        fontweight="bold",
    )
    plt.yticks(
        [145, 155, 165, 170],
        fontsize=10,
        fontweight="bold",
    )
    plt.grid(True, color="grey", linewidth=0.3, linestyle="--")
    # plt.title("Inference Time vs Peak Memory Usage on Jetson Orin Nano")
    plt.tight_layout()
    plt.savefig("graph/method3-2.pdf", dpi=1200, bbox_inches="tight")


draw1()
# draw2()
