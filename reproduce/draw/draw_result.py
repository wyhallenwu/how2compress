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

# Data from your table including Uniform QP
data = {
    "Video": ["MOT17-02", "MOT17-04", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"],
    "Uniform_QP_AS": [84.5, 49.5, 83.4, 47.7, 57.7, 54.9],
    "Uniform_QP_mAP50_95": [0.674, 0.624, 0.673, 0.542, 0.666, 0.522],
    "Spatial_AQ_AS": [77.2, 45.4, 74.1, 43.9, 51.8, 50.4],
    "Spatial_AQ_mAP50_95": [0.672, 0.621, 0.671, 0.540, 0.665, 0.517],
    "AccMPEG_AS": [68.9, 46.2, 72.0, 45.5, 53.2, 52.7],
    "AccMPEG_mAP50_95": [0.673, 0.621, 0.671, 0.541, 0.665, 0.524],
    "Adaptive_QP_AS": [60.8, 40.8, 50.6, 34.6, 41.5, 37.0],
    "Adaptive_QP_mAP50_95": [0.671, 0.620, 0.670, 0.540, 0.661, 0.518],
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
        "Compression_Rate": compression_rates["Uniform_QP"]
        + compression_rates["Spatial_AQ"]
        + compression_rates["AccMPEG"]
        + compression_rates["Adaptive_QP"],
        "mAP50_95": data["Uniform_QP_mAP50_95"]
        + data["Spatial_AQ_mAP50_95"]
        + data["AccMPEG_mAP50_95"]
        + data["Adaptive_QP_mAP50_95"],
        "Method": ["Uniform QP"] * 6
        + ["Spatial AQ Auto"] * 6
        + ["AccMPEG"] * 6
        + ["Adaptive QP"] * 6,
    }
)

# Set up the matplotlib figure for ACM two-column format with desired size
fig, axs = plt.subplots(1, len(data["Video"]), figsize=(11, 2), sharey=True)

# Define markers for each method
markers = {
    "Uniform QP": "D",
    "Spatial AQ Auto": "o",
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
    "Spatial AQ Auto": "#E84446",  # Pink
    "AccMPEG": "#59B78F",  # Green
    "Adaptive QP": "#7A378A",  # Blue
}
new_category_names = {
    "Uniform QP": "Uni.",
    "Spatial AQ Auto": "S.AQ",
    "AccMPEG": "AccMPEG",
    "Adaptive QP": "Ours",
}

# Plot each video in a separate subplot
for i, video in enumerate(data["Video"]):
    sns.scatterplot(
        x="Compression_Rate",
        y="mAP50_95",
        hue="Method",
        style="Method",
        markers=markers,
        palette=colors,
        data=df[df["Video"] == video],
        ax=axs[i],
        s=65,
        edgecolor="black",  # Set edge color to black
        linewidth=1,  # Set edge width
    )
    # axs[i].set_title(f"{video}", fontsize=8)
    axs[i].set_xlabel("Compression Rate", fontsize=10)
    axs[i].set_ylabel("mAP50-95", fontsize=10) if i == 0 else axs[i].set_ylabel("")
    axs[i].tick_params(axis="x", labelsize=8)

    axs[i].tick_params(axis="y", labelsize=8)
    axs[i].set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    axs[i].set_xticklabels(["0", "0.1", "0.2", "0.3", "0.4"])

    # Set specific y-axis ticks
    axs[i].set_yticks([0.3, 0.5, 0.9])
    axs[i].set_yticklabels(["0.3", "0.5", "0.9"])
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
        0.7,
        0.2,  # Adjust these coordinates to position the text near the arrowhead
        "Better",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
        transform=axs[i].transAxes,
    )

    # Make the legend smaller
    # axs[i].legend(fontsize=5, title_fontsize=6)
    # Adjust the legend to have two columns and two rows
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
    )


# Adjust the legend for the whole figure if needed
handles, labels = axs[-1].get_legend_handles_labels()
# fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=6, frameon=False)

# Adjust layout to fit ACM two-column format
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
# plt.subplots_adjust(top=0.8)  # Make space for the suptitle

# # Show the plot
# plt.suptitle(
#     "Compression Rate vs mAP50-95 for Different Video Clips", y=1.0, fontsize=8
# )

plt.savefig("graph/result1.pdf", dpi=1200)
