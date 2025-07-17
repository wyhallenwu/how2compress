import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import pandas as pd


result = pd.read_csv(
    # "/how2compress/results/MOT17-04eval30-45-1713.txt",
    # "/how2compress/results/MOT17-04eval30-45-1704.txt",
    # "/how2compress/results/MOT17-09eval30-45-1709-2.txt",
    # "/how2compress/results/MOT17-04eval30-45-accmpeg-1709.txt",
    # "/how2compress/results/MOT17-13eval30-45-accmpeg-1713-1.txt",
    "/how2compress/results/eval30-45-MOT17-02-f7-1.txt",
    header=None,
    names=[
        "mAPs50_95",
        "mAPs75",
        "mAPs50",
        "mAPs_gt50_95",
        "mAPs_gt75",
        "mAPs_gt50",
        "frames_size",
        "times",
    ],
)


result2 = pd.read_csv(
    # "/how2compress/results/MOT17-04eval30-45-1713.txt",
    # "/how2compress/results/MOT17-04eval30-45-1704.txt",
    # "/how2compress/results/MOT17-09eval30-45-1709-2.txt",
    # "/how2compress/results/MOT17-04eval30-45-accmpeg-1709.txt",
    # "/how2compress/results/MOT17-13eval30-45-accmpeg-1713-1.txt",
    # "/how2compress/results/eval30-45-MOT17-02-f7-1.txt",
    "/how2compress/results/MOT17-09eval30-45-accmpeg-1702.txt",
    header=None,
    names=[
        "mAPs50_95",
        "mAPs75",
        "mAPs50",
        "mAPs_gt50_95",
        "mAPs_gt75",
        "mAPs_gt50",
        "frames_size",
        "times",
    ],
)

# Example data: a cluster of points
points = result[["times", "mAPs50_95"]].to_numpy()
points2 = result2[["times", "mAPs50_95"]].to_numpy()

# Calculate the mean of the points
mean = np.mean(points, axis=0)
mean2 = np.mean(points2, axis=0)

# Calculate the covariance matrix of the points
cov = np.cov(points, rowvar=False)
cov2 = np.cov(points2, rowvar=False)

# Compute the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(cov)
eigenvalues2, eigenvectors2 = np.linalg.eigh(cov2)

# Sort the eigenvalues and eigenvectors by descending order of eigenvalues
order = eigenvalues.argsort()[::-1]
order2 = eigenvalues2.argsort()[::-1]
eigenvalues = eigenvalues[order]
eigenvalues2 = eigenvalues2[order2]
eigenvectors = eigenvectors[:, order]
eigenvectors2 = eigenvectors2[:, order2]

# Compute the angle of the ellipse
angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
angle2 = np.degrees(np.arctan2(*eigenvectors2[:, 0][::-1]))

# Width and height of the ellipse are 2*sqrt(eigenvalue) (for a 95% confidence interval, use a larger factor)
width, height = 2 * np.sqrt(eigenvalues)
width2, height2 = 2 * np.sqrt(eigenvalues2)

# Create the ellipse patch
ellipse = Ellipse(
    xy=mean, width=width, height=height, angle=angle, edgecolor="red", facecolor="none"
)
ellipse2 = Ellipse(
    xy=mean2,
    width=width2,
    height=height2,
    angle=angle2,
    edgecolor="blue",
    facecolor="none",
)
# Plot the points and the ellipse
fig, ax = plt.subplots()
# ax.scatter(points[:, 0], points[:, 1])
ax.add_patch(ellipse)
ax.add_patch(ellipse2)

# Plot the mean point
ax.scatter(mean[0], mean[1], color="red", marker="x")

# Set limits and labels
ax.set_xlim(min(points[:, 0]) - 0.2, max(points[:, 0]) + 0.2)
ax.set_ylim(min(points[:, 1]) - 0.2, max(points[:, 1]) + 0.2)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
# Show plot
plt.savefig("test.png")
