import os
import subprocess
import numpy as np

# Define the directory where the files are stored
directory = "/how2compress/video-result"
dataset = "MOT17-02"

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
ours_sizes = []
accmpeg_sizes = []
uni_sizes = []
aq_sizes = []


def get_iframe_sizes(video_file):
    # Construct the ffprobe command
    command = [
        "ffprobe",
        "-select_streams",
        "v",
        "-show_frames",
        "-show_entries",
        "frame=pkt_size,pict_type",
        "-of",
        "csv",
        video_file,
    ]

    # Run the command and capture the output
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )

    # Filter for I-frames (lines containing "I")
    output_lines = result.stdout.splitlines()

    for line in output_lines:
        if ",I" in line:
            # Extract the size from the CSV output
            frame_data = line.split(",")
            size = int(frame_data[1])  # Assuming pkt_size is the second column
            return size


for file in ours_files:
    frame_sizes = get_iframe_sizes(file)
    ours_sizes.append(frame_sizes)

for file in accmpeg_files:
    frame_sizes = get_iframe_sizes(file)
    accmpeg_sizes.append(frame_sizes)

for file in uni_files:
    frame_sizes = get_iframe_sizes(file)
    uni_sizes.append(frame_sizes)

for file in aq_files:
    frame_sizes = get_iframe_sizes(file)
    aq_sizes.append(frame_sizes)


print(dataset)
print(f"average sizes of I-frames in ours: {np.mean(ours_sizes)}")
print(f"average sizes of I-frames in accmpeg: {np.mean(accmpeg_sizes)}")
print(f"average sizes of I-frames in uni: {np.mean(uni_sizes)}")
print(f"average sizes of I-frames in aq: {np.mean(aq_sizes)}")
