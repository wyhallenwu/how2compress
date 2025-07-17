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
font_path = "/usr/share/fonts/truetype/msttcorefonts/arialbd.ttf"
prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
font1_path = "/usr/share/fonts/truetype/msttcorefonts/Verdana_Bold_Italic.ttf"
fm.fontManager.addfont(font1_path)
verdanabi= fm.FontProperties(fname=font1_path)

plt.rcParams["font.family"] = prop.get_name()
plt.rcParams["font.weight"] = "bold"

def draw_motivations2():
    data = {
        "Dataset": [
            "Scene: A",
            "Scene: B",
            "Scene: C",
            "Scene: D",
        ],
        "AS": [84.5, 49.5, 83.4, 47.7],
        "FL": [79.5, 45.5, 76.5, 44.1],
        "AQ": [77.2, 45.4, 74.1, 43.9],
    }
    df = pd.DataFrame(data)
    df["Frame-level"] = (1 - df["FL"] / df["AS"]) * 100
    df["Macroblock-level"] = (1 - df["AQ"] / df["AS"]) * 100
    df["Improvement"] = df["Frame-level"] - df["Macroblock-level"]
    df_melted = df.melt(
        id_vars="Dataset",
        value_vars=["Frame-level", "Macroblock-level"],
        var_name="Metric",
        value_name="Percentage",
    )
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.weight"] = "bold"
    plt.figure(figsize=(3.4, 2))  # Adjusted figsize
    sns.barplot(
        x="Dataset",
        y="Percentage",
        hue="Metric",
        data=df_melted,
        errorbar=None,
        palette=["#6888F5", "#D77071"],
        edgecolor="black",
        linewidth=1,  # Adjusted linewidth
    )
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    for i, dataset in enumerate(df["Dataset"]):
        macroblock_y = df.loc[i, "Macroblock-level"]
        frame_y = df.loc[i, "Frame-level"]
        improvement = df.loc[i, "Improvement"]
        plt.plot(
            [i - 0.2, i + 0.2],
            [macroblock_y, macroblock_y],
            color="black",
            linestyle="--",
        )
        plt.plot([i - 0.2, i + 0.2], [frame_y, frame_y], color="black", linestyle="--")
        plt.annotate(
            "",
            xy=(i - 0.1, frame_y),
            xytext=(i - 0.1, macroblock_y),
            arrowprops=dict(
                facecolor="black", shrink=0.05, width=0.2, headwidth=3, headlength=3
            ),  # Adjusted arrowprops
        )
        plt.text(
            i - 0.35,
            (frame_y + macroblock_y) / 2,
            f"{abs(improvement):.1f}%",
            ha="center",
            va="center",
            fontsize=7,  # Adjusted fontsize
            color="black",
        )
    plt.xlabel("", fontsize=4, fontweight="bold")  # Adjusted fontsize
    plt.ylabel("Compression Rate", fontsize=8, fontweight="bold")  # Adjusted fontsize
    plt.xticks(fontsize=7)  # Adjusted fontsize
    plt.yticks(fontsize=6)  # Adjusted fontsize
    plt.legend(title="", fontsize=6, title_fontsize=6)  # Adjusted fontsize
    plt.ylim(5.5, 12)
    plt.savefig("motivation2-compress.pdf")


def draw_motivation2_1():
    data = {
        "Dataset": [
            "Scene: B",
            "Scene: D",
        ],
        "AS": [49.5, 47.7],
        "AQ": [45.4, 43.9],
        "CG": [46.2, 45.5],
    }
    df = pd.DataFrame(data)
    df["Fine-grained"] = (1 - df["AQ"] / df["AS"]) * 100
    df["Coarse-grained"] = (1 - df["CG"] / df["AS"]) * 100
    df["Improvement"] = df["Fine-grained"] - df["Coarse-grained"]
    df_melted = df.melt(
        id_vars="Dataset",
        value_vars=["Fine-grained", "Coarse-grained"],
        var_name="Metric",
        value_name="Percentage",
    )
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.weight"] = "bold"
    plt.figure(figsize=(2, 2))  # Adjusted figsize
    sns.barplot(
        x="Dataset",
        y="Percentage",
        hue="Metric",
        data=df_melted,
        errorbar=None,
        palette=["#6888F5", "#D77071"],
        edgecolor="black",
        linewidth=1,  # Adjusted linewidth
    )
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    for i, dataset in enumerate(df["Dataset"]):
        macroblock_y = df.loc[i, "Fine-grained"]
        frame_y = df.loc[i, "Coarse-grained"]
        improvement = df.loc[i, "Improvement"]
        plt.plot(
            [i - 0.2, i + 0.2],
            [macroblock_y, macroblock_y],
            color="black",
            linestyle="--",
        )
        plt.plot([i - 0.2, i + 0.2], [frame_y, frame_y], color="black", linestyle="--")
        plt.annotate(
            "",
            xy=(i - 0.1, frame_y),
            xytext=(i - 0.1, macroblock_y),
            arrowprops=dict(
                facecolor="black", shrink=0.05, width=0.2, headwidth=3, headlength=3
            ),  # Adjusted arrowprops
        )
        plt.text(
            i - 0.25,
            (frame_y + macroblock_y) / 2,
            f"{abs(improvement):.1f}%",
            ha="center",
            va="center",
            fontsize=7,  # Adjusted fontsize
            color="black",
        )
    plt.xlabel("", fontsize=4, fontweight="bold")  # Adjusted fontsize
    plt.ylabel("", fontsize=8, fontweight="bold")  # Adjusted fontsize
    plt.xticks(fontsize=7)  # Adjusted fontsize
    plt.yticks(fontsize=6)  # Adjusted fontsize
    plt.legend(title="", fontsize=6, title_fontsize=6)  # Adjusted fontsize
    plt.ylim(3, 10)
    plt.savefig("motivation2-compress-2.pdf")


def draw_motivations():
    bg = {
        "QP": [30, 35, 40, 45, 50],
        "mAP50-95": [0.624, 0.593, 0.582, 0.562, 0.539],
        "mAP75": [0.708, 0.681, 0.668, 0.645, 0.619],
        "mAP50": [0.775, 0.746, 0.734, 0.712, 0.695],
    }
    roi = {
        "QP": [30, 35, 40, 45, 50],
        "mAP50-95": [0.624, 0.591, 0.560, 0.505, 0.444],
        "mAP75": [0.708, 0.677, 0.640, 0.569, 0.507],
        "mAP50": [0.775, 0.746, 0.717, 0.658, 0.584],
    }

    # Convert dictionaries to DataFrames
    bg_df = pd.DataFrame(bg)
    roi_df = pd.DataFrame(roi)

    # Normalizing the values to [0,1] with respect to QP 30's column
    bg_df_normalized = bg_df.copy()
    roi_df_normalized = roi_df.copy()

    for column in bg_df.columns[1:]:
        bg_df_normalized[column] = bg_df[column] / bg_df[column].iloc[0]
        roi_df_normalized[column] = roi_df[column] / roi_df[column].iloc[0]

    # Adding the 'Category' column to identify 'bg' and 'roi'
    bg_df_normalized["Category"] = "Mask BG"
    roi_df_normalized["Category"] = "Mask RoI"

    # Concatenating the dataframes
    combined_df = pd.concat([bg_df_normalized, roi_df_normalized])

    # Melting the DataFrame for seaborn
    melted_df = pd.melt(
        combined_df,
        id_vars=["QP", "Category"],
        var_name="Metric",
        value_name="Normalized Value",
    )
    print(melted_df)

    # Plotting the bar graph with hues
    plt.figure(figsize=(3.3, 2.9))
    plt.rcParams["font.family"] = "Arial"
    # plt.rcParams["font.weight"] = "bold"
    ax = sns.barplot(
        data=melted_df,
        x="QP",
        y="Normalized Value",
        hue="Category",
        errorbar=None,
        # palette=["#E84446", "#6A8EC9"],
        palette=["#72bb6c", "#fcaa57"],
        edgecolor="black",
        linewidth=2,
        alpha=1,
    )
    # plt.title("Normalized mAP Values for BG and ROI", fontsize=16)
    plt.xlabel("Quantization Parameter", fontsize=12)
    plt.ylabel("Normalized mAP50-95", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title="", fontsize=10, title_fontsize=10)
    plt.ylim(0.7, 1)

    # hatches = {"Mask BG": "//", "Mask RoI": "\\\\"}

    # Apply hatches to the bars
    hatches = ["xx", "-"]  # Hatch patterns for BG and RoI
    bars = ax.patches  # Get all the bars

    for i, bar in enumerate(bars):
        # Alternate between the hatches based on the hue (Category)
        bar.set_hatch(hatches[0 if i < (len(bars) // 2 - 1) else 1])  # Apply hatch
    bg_patch = Rectangle(
        (0, 0), 1, 1, facecolor="#72bb6c", edgecolor="black", linewidth=1, hatch="xx"
    )
    roi_patch = Rectangle(
        (0, 0), 1, 1, facecolor="#fcaa57", edgecolor="black", linewidth=1, hatch="-"
    )
    plt.legend(
        [bg_patch, roi_patch],
        ["Mask BG", "Mask RoI"],
        title="",
        fontsize=10,
        title_fontsize=10,
        handlelength=1.5,
        handleheight=1,
        loc="upper right",
    )

    plt.tight_layout()
    plt.savefig("graph/motivation-mAP50-95.pdf", dpi=2400)


def draw_3dbar():
    # Fixing random state for reproducibility
    np.random.seed(19680801)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    x, y = np.random.rand(2, 100) * 4
    hist, xedges, yedges = np.histogram2d(x, y, bins=4, range=[[0, 4], [0, 4]])

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Set dimensions for the bars to eliminate the space between them
    dx = dy = 0.9
    dz = hist.ravel() * 4

    ax.bar3d(
        xpos,
        ypos,
        zpos,
        dx,
        dy,
        dz,
        zsort="average",
        color="#6db1ff",
        edgecolor="black",
        linewidth=0.8,
    )

    # Add value annotations on top of each bar

    # for i in range(len(xpos)):
    #     ax.text(
    #         xpos[i] + dx / 2,
    #         ypos[i] + dy / 2,
    #         zpos + dz[i] * 0.1,
    #         "%d" % dz[i],
    #         ha="center",
    #         va="bottom",
    #         fontsize=13,
    #         color="black",
    #     )

    # Remove the background
    ax.set_box_aspect([1, 1, 0.5])  # Aspect ratio is 1:1:0.5

    # Create a bottom plane separated from the bar plot
    X, Y = np.meshgrid(
        np.arange(-1, 5, 1), np.arange(-1, 5, 1)
    )  # Extend the plane slightly beyond the bars
    Z = np.zeros_like(X) - 25  # Position the plane below the bars

    ax.plot_surface(X, Y, Z, color="#cfe4ff", alpha=1)

    # Make the panes transparent
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Hide the axes and grid lines
    ax.grid(False)
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Hide x axis line
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Hide y axis line
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Hide z axis line
    # Remove all text (axis labels, tick marks, tick labels)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    plt.tight_layout()
    plt.savefig("3dbar.png")


def draw_combined_motivation():
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
        1, 2, figsize=(4.5, 2), gridspec_kw={"width_ratios": [0.6, 0.4]}
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
            linestyle="--",
        )
        ax1.plot([i - 0.2, i + 0.2], [frame_y, frame_y], color="black", linestyle="--")
        ax1.annotate(
            "",
            xy=(i - 0.1, frame_y),
            xytext=(i - 0.1, macroblock_y),
            arrowprops=dict(
                facecolor="black", shrink=0.05, width=0.2, headwidth=3, headlength=3
            ),
        )
        ax1.text(
            i - 0.25,
            (frame_y + macroblock_y) / 2,
            f"{abs(improvement):.1f}%",
            ha="center",
            va="center",
            fontsize=7,
            color="black",
            # fontweight="bold",
        )

    ax1.set_xlabel("", fontsize=4, fontweight="bold")
    ax1.set_ylabel("Bitrate Saving (Raw QP 25)", fontsize=10, fontweight="bold")
    ax1.set_xticks(ax1.get_xticks())
    ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=7)
    ax1.set_yticks([50, 55, 60, 65, 70])
    ax1.set_yticklabels([f"{int(y)}%" for y in ax1.get_yticks()], fontsize=8, rotation=90, fontweight="bold")
    # print(f"ax1.get_xticks(): {ax1.get_xticks()}")
    # print(f"ax1.get_yticks(): {ax1.get_yticks()}")

    ax1.legend(title="", fontsize=6, title_fontsize=6)
    ax1.set_ylim(50, 70)

    # Apply hatches to the bars
    hatches = ["xx", "--"]  # Hatch patterns for BG and RoI
    bars = ax1.patches  # Get all the bars

    for i, bar in enumerate(bars):
        # Alternate between the hatches based on the hue (Category)
        # bar.set_hatch(hatches[0 if i < (len(bars) // 2 - 1) else 1])  # Apply hatch
        bar.set_hatch(hatches[0 if i < (len(bars) // 2 - 1) else 1])
        bar.set_edgecolor("white")
        bar.set_linewidth(0.1)

    fl_patch = Rectangle(
        (0, 0), 1, 1, facecolor="#72bb6c", edgecolor="white", linewidth=0.1, hatch="xx"
    )
    mb_patch = Rectangle(
        (0, 0), 1, 1, facecolor="#fcaa57", edgecolor="white", linewidth=0.1, hatch="--"
    )
    ax1.legend(
        [fl_patch, mb_patch],
        ["Frame-Level(when2compress)", "Macroblock-Level(Codec's AQ)"], # custom here
        title="",
        fontsize=6,
        title_fontsize=6,
        handlelength=1.5,
        handleheight=1,
        loc="upper right",
    )

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
            linestyle="--",
        )
        ax2.plot([i - 0.2, i + 0.2], [frame_y, frame_y], color="black", linestyle="--")
        ax2.annotate(
            "",
            xy=(i - 0.1, frame_y),
            xytext=(i - 0.1, macroblock_y),
            arrowprops=dict(
                facecolor="black", shrink=0.05, width=0.2, headwidth=3, headlength=3
            ),
        )
        ax2.text(
            i - 0.25,
            (frame_y + macroblock_y) / 2,
            f"{abs(improvement):.1f}%",
            ha="center",
            va="center",
            fontsize=7,
            color="black",
            fontweight="bold",
        )

    ax2.set_xlabel("", fontsize=4, fontweight="bold")
    ax2.set_ylabel("", fontsize=8, fontweight="bold")
    ax2.set_xticks(ax2.get_xticks())
    ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=7)
    # ax2.set_yticks(ax2.get_yticks())
    ax2.set_yticks([])
    # ax2.set_yticklabels([f"{int(y)}%" for y in ax2.get_yticks()], fontsize=6)

    ax2.legend(title="", fontsize=6, title_fontsize=6)
    ax2.set_ylim(50, 70)

    # Apply hatches to the bars
    hatches = ["++", "**"]  # Hatch patterns for BG and RoI
    bars = ax2.patches  # Get all the bars

    for i, bar in enumerate(bars):
        # Alternate between the hatches based on the hue (Category)
        # bar.set_hatch(hatches[0 if i < (len(bars) // 2 - 1) else 1])  # Apply hatch
        bar.set_hatch(hatches[0 if i < (len(bars) // 2 - 1) else 1])
        bar.set_edgecolor("white")
        bar.set_linewidth(0.1)
    cg_patch = Rectangle(
        (0, 0), 1, 1, facecolor="#F38276", edgecolor="white", linewidth=0.1, hatch="++"
    )
    finel_patch = Rectangle(
        (0, 0), 1, 1, facecolor="#91CAE8", edgecolor="white", linewidth=0.1, hatch="**"
    )
    ax2.legend(
        [cg_patch, finel_patch],
        ["Coarse(where2compress)", "Fine(Codec's AQ)"],
        title="",
        fontsize=6,
        title_fontsize=10,
        handlelength=1.5,
        handleheight=1,
        loc="upper right",
    )

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.2)  # Increase or decrease this value as needed

    # Save combined figure
    plt.savefig(
        "graph/combined_motivation.pdf", dpi=2400, bbox_inches="tight", pad_inches=0.1
    )
    # plt.show()


# draw_combined_motivation()


if __name__ == "__main__":
    # draw_motivations()
    # draw_3dbar()
    # draw_motivations2()
    # draw_motivation2_1()
    draw_combined_motivation()
