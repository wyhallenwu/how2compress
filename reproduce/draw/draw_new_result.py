import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set default font weight to bold
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.weight"] = "bold"
# plt.rcParams["axes.labelweight"] = "bold"
# plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["xtick.labelsize"] = 6
plt.rcParams["ytick.labelsize"] = 6
# plt.rcParams["xtick.labelweight"] = "bold"
# plt.rcParams["ytick.labelweight"] = "bold"
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["legend.title_fontsize"] = 10
plt.rcParams["legend.frameon"] = True


df = pd.read_csv("/how2compress/results/csv/result.csv")

data = {
    "Video": ["MOT17-02", "MOT17-04", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"],
    "Uniform_QP_AS": [4075.52, 1646.07, 3985.58, 5026.91, 5116.90, 3711.07],
    "Uniform_QP_mAP50": [
        0.8418439636108657,
        0.8514149107482453,
        0.8571097551804281,
        0.847342368864796,
        0.887071137029182,
        0.8244032882549087,
    ],
    "Spatial_AQ_AS": [2914.09, 1361.96, 3258.17, 3877.78, 4100.88, 3144.42],
    "Spatial_AQ_mAP50": [
        0.8385749023101287,
        0.8470506393130917,
        0.8540700358326323,
        0.8444614706867449,
        0.885727668809539,
        0.8152140770570216,
    ],
    "AccMPEG_AS": [3353.73, 1569.07, 3622.47, 4517.54, 4468.73, 3300.35],
    "AccMPEG_mAP50": [
        0.8445474044983011,
        0.8489860867325955,
        0.8585917971697611,
        0.8426418522395146,
        0.8872131819346438,
        0.8178748444335404,
    ],
    "Adaptive_QP_AS": [2022.78, 1388.04, 2942.78, 3468.95, 3487.50, 2759.26],
    "Adaptive_QP_mAP50": [
        0.8367827290036991,
        0.846493480283848,
        0.851300964912706,
        0.8378676684446574,
        0.8739444567032358,
        0.8075526429972927,
    ],
}

# Calculate the compression rates relative to Uniform QP 30
compression_rates = {
    "Uniform_QP": [1.0] * len(data["Video"]),  # Baseline, so compression rate is 1
    "Spatial_AQ": [x / y for x, y in zip(data["Spatial_AQ_AS"], data["Uniform_QP_AS"])],
    "AccMPEG": [x / y for x, y in zip(data["AccMPEG_AS"], data["Uniform_QP_AS"])],
    "Adaptive_QP": [
        x / y for x, y in zip(data["Adaptive_QP_AS"], data["Uniform_QP_AS"])
    ],
}

compression_rates = {
    method: [1 - val for val in values] for method, values in compression_rates.items()
}

# Create a DataFrame for easier plotting with Seaborn
df = pd.DataFrame(
    {
        "Video": data["Video"] * 4,
        "Relative Bitrate": compression_rates["Uniform_QP"]
        + compression_rates["Spatial_AQ"]
        + compression_rates["AccMPEG"]
        + compression_rates["Adaptive_QP"],
        "mAP50": data["Uniform_QP_mAP50"]
        + data["Spatial_AQ_mAP50"]
        + data["AccMPEG_mAP50"]
        + data["Adaptive_QP_mAP50"],
        "Method": ["Uniform QP"] * 6
        + ["AQ"] * 6
        + ["AccMPEG"] * 6
        + ["Adaptive QP"] * 6,
    }
)

print(df)

# Set up the matplotlib figure for ACM two-column format with desired size
fig, axs = plt.subplots(1, len(data["Video"]), figsize=(11, 1.9), sharey=True)

# Define markers for each method
markers = {
    "Uniform QP": "D",
    "AQ": "o",
    "AccMPEG": "s",
    "Adaptive QP": "p",
}
# colors = {
#     "Uniform QP": "deepskyblue",
#     "Spatial AQ Auto": "green",
#     "AccMPEG": "orange",
#     "Adaptive QP": "red",
# }
colors = {
    "Uniform QP": "#6A8EC9",  # Purple
    "AQ": "#E84446",  # Pink
    "AccMPEG": "#59B78F",  # Green
    "Adaptive QP": "#7A378A",  # Blue
}
new_category_names = {
    "Uniform QP": "Uni.",
    "AQ": "AQ",
    "AccMPEG": "AccMPEG",
    "Adaptive QP": "Ours",
}

# Plot each video in a separate subplot
for i, video in enumerate(data["Video"]):
    sns.scatterplot(
        x="Relative Bitrate",
        y="mAP50",
        hue="Method",
        style="Method",
        markers=markers,
        palette=colors,
        data=df[df["Video"] == video],
        ax=axs[i],
        s=100,
        edgecolor="black",  # Set edge color to black
        linewidth=1,  # Set edge width
    )
    # axs[i].set_title(f"{video}", fontsize=8)
    axs[i].set_xlabel("Bitrate Saving", fontsize=12)
    axs[i].set_ylabel("mAP50", fontsize=12) if i == 0 else axs[i].set_ylabel("")
    axs[i].tick_params(axis="x", labelsize=10)

    axs[i].tick_params(axis="y", labelsize=10)
    axs[i].set_xticks([-0.04, 0, 0.2, 0.4, 0.6])
    axs[i].set_xticklabels(["", "0", "20%", "40%", "60%"])

    # Set specific y-axis ticks
    axs[i].set_yticks([0.7, 0.9, 1])
    axs[i].set_yticklabels(["0.7", "0.9", ""])
    axs[i].grid(True, color="grey", linestyle="--", linewidth=0.3)
    axs[i].legend(fontsize=14, prop={"weight": "bold"})
    # Add the 'Better' annotation with an arrow
    axs[i].annotate(
        "",  # Remove text from here
        xy=(0.9, 0.2),  # End point of the arrow (upper right)
        xytext=(0.6, 0),  # Start point of the arrow (lower left)
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(
            facecolor="black",
            edgecolor="black",
            arrowstyle="->",
            linewidth=1.5,
            connectionstyle="arc3,rad=0.2",
        ),
    )

    # Add text near arrowhead
    axs[i].text(
        0.6,
        0.2,  # Adjust these coordinates to position the text near the arrowhead
        "Better",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
        transform=axs[i].transAxes,
    )

    handles, labels = axs[i].get_legend_handles_labels()
    labels = [new_category_names[label] for label in labels]  # Update labels
    axs[i].legend(
        handles=handles,
        labels=labels,
        fontsize=6,
        # title_fontsize=6,
        # loc="upper center",
        # bbox_to_anchor=(0.5, 1.35),
        ncol=2,
        frameon=True,
        framealpha=1,
        edgecolor="grey",
        facecolor="white",
        # prop={"weight": "bold"},
    )


# Adjust layout to fit ACM two-column format
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
# plt.subplots_adjust(right=0.8)  # Make space for the suptitle
plt.savefig("graph/new_result1.pdf", dpi=2400)
