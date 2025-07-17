import csv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import re
from matplotlib import gridspec

# filename1 = (
#     "/how2compress/results/decisions30-45-accmpeg-MOT17-04-paper-0.5q.txt"
# )
# filename2 = "/how2compress/results/decisions30-45-MOT17-04-f7.txt"
frame = "/how2compress/data/MOT17Det/train/MOT17-10/img1/000001.jpg"

filename1 = "/how2compress/results/csv/1713-01-accmpeg.csv"
filename2 = "/how2compress/results/csv/1713-01-ours.csv"

filename3 = "/how2compress/results/csv/1713-01-aq-nv.csv"


filename4 = "/how2compress/results/csv/1710-26-accmpeg.csv"
filename5 = "/how2compress/results/csv/1710-26-ours.csv"


def parse(file: str):
    # Define a function to process each line
    def process_line(line):
        line_cleaned = re.sub(r"\[.*?\] ", "", line.strip())
        decimal_pairs = [
            line_cleaned[i : i + 2] for i in range(0, len(line_cleaned), 2)
        ]

        values = [int(dp) for dp in decimal_pairs]

        return values

    with open(file, "r") as file:
        matrix = [process_line(line) for line in file.readlines()]

    return np.array(matrix)


def draw(filename: str, frame: str, ax):
    data = parse(filename)
    # plt.figure(figsize=(4, 3))
    data = data.reshape(68, 120)
    # colors = ["#150F89", "#6900A7", "#CC4973", "#F8953F", "#F0F224"]
    # cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    # custom_colors = ["#6A8EC9", "#E84446"]
    # custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", custom_colors)

    heatmap = sns.heatmap(
        data,
        cmap="coolwarm",
        cbar=False,
        annot=False,
        alpha=1,
        zorder=2,
        linewidths=0,
        ax=ax,
    )
    # Access the colorbar object
    # colorbar = heatmap.collections[0].colorbar
    bg = Image.open(frame)
    bg = bg.resize((120, 68))

    ax.imshow(bg, aspect="auto", zorder=1)

    # # Set custom ticks for the colorbar
    # custom_ticks = [0, 1, 2, 3, 4]  # Define the tick positions
    # colorbar.set_ticks(custom_ticks)  # Apply the custom ticks
    # colorbar.set_ticklabels(["", "", "", "", ""])  # Apply custom labels
    # colorbar.ax.tick_params(left=False, right=False, labelleft=False, labelright=True)
    # # Optional: Customize the colorbar label and tick size
    # colorbar.set_label("Emphasis Level", fontsize=14)
    # colorbar.ax.yaxis.set_tick_params(labelsize=10)
    ax.axis("off")
    # ax.tight_layout()
    # plt.savefig("graph/decisions_ours-1704.pdf")
    # plt.close()


def draw_single_row2(fileaname1: str, filename2: str, frame: str):
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.weight"] = "bold"
    figure = plt.figure(figsize=(6.6, 1.8))
    gs = gridspec.GridSpec(
        1, 3, width_ratios=[32, 32, 2]
    )  # Grid layout: 30:1 ratio for heatmap and colorbar
    ax_main1 = plt.subplot(gs[0, 0])

    data1 = parse(fileaname1)
    data1 = data1.reshape(68, 120)
    heatmap = sns.heatmap(
        data1,
        cmap="coolwarm",
        cbar=False,
        annot=False,
        alpha=0.6,
        zorder=2,
        linewidths=0,
        ax=ax_main1,
    )
    bg = Image.open(frame)
    bg = bg.resize((120, 68))
    ax_main1.imshow(bg, aspect="auto", zorder=1)
    ax_main1.axis("off")

    data2 = parse(filename2)
    data2 = data2.reshape(68, 120)
    ax_main2 = plt.subplot(gs[0, 1])
    heatmap = sns.heatmap(
        data2,
        cmap="coolwarm",
        cbar=False,
        annot=False,
        alpha=1,
        zorder=2,
        linewidths=0,
        ax=ax_main2,
    )
    bg = Image.open(frame)
    bg = bg.resize((120, 68))
    ax_main2.imshow(bg, aspect="auto", zorder=1)
    ax_main2.axis("off")

    # Create the colorbar axis
    cbar_ax = plt.subplot(gs[0, 2])  # The second axis in the grid
    cbar = plt.colorbar(
        heatmap.get_children()[0], cax=cbar_ax
    )  # Manually link the colorbar to the heatmap
    # plt.axis("off")
    cbar.set_ticks([np.min(data1), np.max(data1)])
    cbar.set_ticklabels([str(np.min(data1)), str(np.max(data1))])
    cbar.ax.tick_params(labelsize=10, length=0)
    cbar.set_label("QP", fontsize=12, weight="bold", labelpad=-3, rotation=270)
    plt.tight_layout()
    plt.savefig("graph/decisions_comp.pdf")


def draw_single1(fileaname1: str, frame: str):
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.weight"] = "bold"
    figure = plt.figure(figsize=(6.6, 1.8))
    gs = gridspec.GridSpec(
        1, 3, width_ratios=[32, 32, 2]
    )  # Grid layout: 30:1 ratio for heatmap and colorbar
    ax_main1 = plt.subplot(gs[0, 0])
    ax_main2 = plt.subplot(gs[0, 1])

    bg = Image.open(frame)
    bg = bg.resize((120, 68))
    ax_main1.imshow(bg, aspect="auto", zorder=1)
    ax_main1.axis("off")

    data1 = parse(fileaname1)
    data1 = data1.reshape(68, 120)
    heatmap = sns.heatmap(
        data1,
        cmap="coolwarm",
        cbar=False,
        annot=False,
        alpha=1,
        zorder=2,
        linewidths=0,
        ax=ax_main2,
    )
    bg = Image.open(frame)
    bg = bg.resize((120, 68))
    ax_main2.imshow(bg, aspect="auto", zorder=1)
    ax_main2.axis("off")

    # Create the colorbar axis
    cbar_ax = plt.subplot(gs[0, 2])  # The second axis in the grid
    cbar = plt.colorbar(
        heatmap.get_children()[0], cax=cbar_ax
    )  # Manually link the colorbar to the heatmap
    # plt.axis("off")
    cbar.set_ticks([np.min(data1), np.max(data1)])
    cbar.set_ticklabels([str(np.min(data1)), str(np.max(data1))])
    cbar.ax.tick_params(labelsize=10, length=0)
    cbar.set_label("QP", fontsize=12, weight="bold", labelpad=-3, rotation=270)
    plt.tight_layout()
    plt.savefig("graph/decisions.pdf")


if __name__ == "__main__":
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.weight"] = "bold"
    # fig, axes = plt.subplots(1, 2, figsize=(6.4, 1.8), sharex=True, sharey=True)
    # draw(filename1, frame, axes[0])
    # draw(filename2, frame, axes[1])

    # # # Create a single colorbar for both heatmaps
    # # cbar_ax = fig.add_axes(
    # #     [0.9, 0.15, 0.02, 0.75]
    # # )  # Adjust position and size as needed
    # # norm = plt.Normalize(vmin=0, vmax=4)
    # # # custom_colors = ["#6A8EC9", "#E84446"]
    # # # custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", custom_colors)
    # # sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    # # sm.set_array([])
    # # cbar = fig.colorbar(sm, cax=cbar_ax)
    # # cbar.set_ticks([0, 4])
    # # cbar.set_ticklabels(["30", "45"])
    # # cbar.ax.tick_params(labelsize=10)
    # # cbar.set_label("QP", fontsize=12, weight="bold")

    # plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to accommodate colorbar
    # plt.savefig("graph/new_decisions.pdf", dpi=1200, bbox_inches="tight")
    # plt.close()

    # draw_single_row2(filename1, filename2, frame)
    draw_single_row2(filename4, filename5, frame)
    # draw_single1(filename3, frame)
