import os
import numpy as np

DATASET = ["MOT17-02", "MOT17-04", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]

for dataset in DATASET:
    directory = "/how2compress/video-result"
    directory1 = "/how2compress/data/UNI25CHUNK"
    directory2 = "/how2compress/data/UNI30CHUNK"
    directory = os.path.join(directory, dataset)
    directory1 = os.path.join(directory1, dataset)
    directory2 = os.path.join(directory2, dataset)

    # Get a list of all files in the directory
    files = os.listdir(directory)
    files1 = os.listdir(directory1)
    files2 = os.listdir(directory2)

    # Filter files that start with 'h264-ours'
    accmpeg = sorted([f for f in files if f.startswith("h264-accmpeg")])
    aq = sorted([f for f in files2 if f.startswith("aq-0")])
    raw = sorted([f for f in files1 if f.startswith("uni")])
    fl = sorted([f for f in files2 if f.startswith("uni")])

    accmpeg_files = [os.path.join(directory, f) for f in accmpeg]
    aq_files = [os.path.join(directory1, f) for f in aq]
    raw_files = [os.path.join(directory1, f) for f in raw]
    fl_files = [os.path.join(directory2, f) for f in fl]

    accmpeg_file_sizes = []
    aq_file_sizes = []
    raw_file_sizes = []
    fl_file_sizes = []

    for file in accmpeg_files:
        accmpeg_file_sizes.append(os.path.getsize(file) / 1024)

    for file in aq_files:
        aq_file_sizes.append(os.path.getsize(file) / 1024)

    for file in raw_files:
        raw_file_sizes.append(os.path.getsize(file) / 1024)

    for file in fl_files:
        fl_file_sizes.append(os.path.getsize(file) / 1024)

    print(f"Dataset: {dataset}")
    print(f"AccMpeg File Sizes: {np.mean(accmpeg_file_sizes)}")
    print(f"AQ File Sizes: {np.mean(aq_file_sizes)}")
    print(f"FL File Sizes: {np.mean(fl_file_sizes)}")
    print(f"Raw File Sizes: {np.mean(raw_file_sizes)}")
    print(
        f"accmpeg compression ratio: { np.mean(accmpeg_file_sizes) / np.mean(raw_file_sizes)}"
    )
    print(f"aq compression ratio: { np.mean(aq_file_sizes) / np.mean(raw_file_sizes)}")
    print(f"fl compression ratio: { np.mean(fl_file_sizes) / np.mean(raw_file_sizes)}")
