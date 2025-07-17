import os
import subprocess
import numpy as np

# Define the directory where the files are stored

DATASET = ["MOT17-02", "MOT17-04", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]

for dataset in DATASET:
    directory = "/how2compress/video-result"
    directory1 = "/how2compress/data/UNI30CHUNK"
    directory = os.path.join(directory, dataset)
    directory1 = os.path.join(directory1, dataset)

    # Get a list of all files in the directory
    files = os.listdir(directory)
    files1 = os.listdir(directory1)

    # Filter files that start with 'h264-ours'
    ours = sorted([f for f in files if f.startswith("h264-ours")])
    accmpeg = sorted([f for f in files if f.startswith("h264-accmpeg")])

    uni = sorted([f for f in files1 if f.startswith("uni")])
    aq = sorted([f for f in files1 if f.startswith("aq")])

    # Join the directory path with the filenames to get full paths
    ours_files = [os.path.join(directory, f) for f in ours]
    accmpeg_files = [os.path.join(directory, f) for f in accmpeg]
    uni_files = [os.path.join(directory1, f) for f in uni]
    aq_files = [os.path.join(directory1, f) for f in aq]

    # ffprobe -v error -select_streams v:0 -show_entries stream=bit_rate -of default=noprint_wrappers=1:nokey=1 input.mp4
    ours_bitrates = []
    accmpeg_bitrates = []
    uni_bitrates = []
    aq_bitrates = []

    for file in ours_files:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=bit_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                file,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        ours_bitrates.append(int(result.stdout.strip()))

    for file in accmpeg_files:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=bit_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                file,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        accmpeg_bitrates.append(int(result.stdout.strip()))

    for file in uni_files:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=bit_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                file,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        uni_bitrates.append(int(result.stdout.strip()))

    for file in aq_files:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=bit_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                file,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        aq_bitrates.append(int(result.stdout.strip()))

    print(dataset)
    print(f"bitrate of ours: {np.mean(ours_bitrates) / 1e3:.2f} Kbps")
    print(f"bitrate of accmpeg: {np.mean(accmpeg_bitrates) / 1e3:.2f} Kbps")
    print(f"bitrate of uni: {np.mean(uni_bitrates) / 1e3:.2f} Kbps")
    print(f"bitrate of aq: {np.mean(aq_bitrates) / 1e3:.2f} Kbps")
