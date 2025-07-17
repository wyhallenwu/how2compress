import os
import subprocess
import numpy as np

# Define the VisDrone validation sequences
DATASETS = [
    "uav0000086_00000_v",
    "uav0000117_02622_v",
    "uav0000137_00458_v",
    "uav0000182_00000_v",
    "uav0000268_05773_v",
    "uav0000305_00000_v",
    "uav0000339_00001_v"
]

for dataset in DATASETS:
    # Define paths for different methods
    results_dir = "/how2compress/results/visdrone"  # For our method and AccMPEG
    uniqp_dir = "/how2compress/data/VisDrone-UNI30-MP4"  # For uniform QP and AQ
    
    # Add _resized suffix to dataset name
    dataset_resized = f"{dataset}_resized"
    
    # Construct full paths
    results_path = os.path.join(results_dir, dataset)
    uniqp_path = os.path.join(uniqp_dir, dataset_resized)

    # Get lists of files for each method
    if os.path.exists(results_path):
        files = os.listdir(results_path)
        # Filter files for our method and AccMPEG
        ours = sorted([f for f in files if f.startswith("ours_chunk")])
        accmpeg = sorted([f for f in files if f.startswith("accmpeg_chunk")])
    else:
        print(f"Warning: Results directory not found: {results_path}")
        ours = []
        accmpeg = []

    if os.path.exists(uniqp_path):
        files1 = os.listdir(uniqp_path)
        # Filter files for uniform QP and AQ
        uni = sorted([f for f in files1 if f.startswith("uni-")])
        aq = sorted([f for f in files1 if f.startswith("aq-1-")])
    else:
        print(f"Warning: UNI/AQ directory not found: {uniqp_path}")
        uni = []
        aq = []

    # Join the directory paths with the filenames to get full paths
    ours_files = [os.path.join(results_path, f) for f in ours]
    accmpeg_files = [os.path.join(results_path, f) for f in accmpeg]
    uni_files = [os.path.join(uniqp_path, f) for f in uni]
    aq_files = [os.path.join(uniqp_path, f) for f in aq]

    # Initialize lists to store bitrates
    ours_bitrates = []
    accmpeg_bitrates = []
    uni_bitrates = []
    aq_bitrates = []

    # Function to get bitrate for a file
    def get_bitrate(file_path):
        try:
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
                    file_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            return int(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            print(f"Error getting bitrate for {file_path}: {e}")
            return 0

    # Get bitrates for each method
    for file in ours_files:
        bitrate = get_bitrate(file)
        if bitrate > 0:
            ours_bitrates.append(bitrate)

    for file in accmpeg_files:
        bitrate = get_bitrate(file)
        if bitrate > 0:
            accmpeg_bitrates.append(bitrate)

    for file in uni_files:
        bitrate = get_bitrate(file)
        if bitrate > 0:
            uni_bitrates.append(bitrate)

    for file in aq_files:
        bitrate = get_bitrate(file)
        if bitrate > 0:
            aq_bitrates.append(bitrate)

    # Print results for this dataset
    print(f"\nDataset: {dataset_resized}")
    if ours_bitrates:
        print(f"Bitrate of ours: {np.mean(ours_bitrates) / 1e3:.2f} Kbps")
    if accmpeg_bitrates:
        print(f"Bitrate of accmpeg: {np.mean(accmpeg_bitrates) / 1e3:.2f} Kbps")
    if uni_bitrates:
        print(f"Bitrate of uniform QP: {np.mean(uni_bitrates) / 1e3:.2f} Kbps")
    if aq_bitrates:
        print(f"Bitrate of AQ: {np.mean(aq_bitrates) / 1e3:.2f} Kbps")

    # Print standard deviations if available
    if len(ours_bitrates) > 1:
        print(f"Std dev of ours: {np.std(ours_bitrates) / 1e3:.2f} Kbps")
    if len(accmpeg_bitrates) > 1:
        print(f"Std dev of accmpeg: {np.std(accmpeg_bitrates) / 1e3:.2f} Kbps")
    if len(uni_bitrates) > 1:
        print(f"Std dev of uniform QP: {np.std(uni_bitrates) / 1e3:.2f} Kbps")
    if len(aq_bitrates) > 1:
        print(f"Std dev of AQ: {np.std(aq_bitrates) / 1e3:.2f} Kbps") 