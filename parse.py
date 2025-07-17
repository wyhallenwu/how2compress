import os
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="dataset name")
parser.add_argument("--result_file", type=str, help="result file")
args = parser.parse_args()

root = "/how2compress/data/MOT17DetH264"

mAPs50_95 = []
mAPs75 = []
mAPs50 = []
mAPs_gt50_95 = []
mAPs_gt75 = []
mAPs_gt50 = []
frames_size = []
times = []

clip = args.dataset
qp = 30
path = os.path.join(root, clip, str(qp))

result = pd.read_csv(
    # "/how2compress/results/MOT17-04eval30-45-1713.txt",
    # "/how2compress/results/MOT17-04eval30-45-1704.txt",
    # "/how2compress/results/MOT17-09eval30-45-1709-2.txt",
    # "/how2compress/results/MOT17-04eval30-45-accmpeg-1709.txt",
    # "/how2compress/results/MOT17-13eval30-45-accmpeg-1713-1.txt",
    args.result_file,
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
# result = result[:450]
result["frames_size"] = result["frames_size"].apply(lambda x: x / 1024)

files = os.listdir(path)
filesizes = [os.path.getsize(os.path.join(path, file)) for file in files]


print(clip)
print(result.mean())
print(result.std())
print(f"avergae file size: {np.mean(filesizes) / 1024}")
# print(result["frames_size"].mean())
print(
    f"compression rate: {100 - result['frames_size'].mean() / np.mean(filesizes) * 1024.0 * 100}%"
)
