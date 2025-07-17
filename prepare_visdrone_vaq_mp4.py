import os
from tqdm import tqdm
import subprocess
import math

# Hardcode all VisDrone validation sequences
DATASETS = [
    "uav0000086_00000_v",
    "uav0000117_02622_v",
    "uav0000137_00458_v",
    "uav0000182_00000_v",
    "uav0000268_05773_v",
    "uav0000305_00000_v",
    "uav0000339_00001_v"
]

# Setup paths
src_root = "/how2compress/data/VisDroneH264VAQ1"
dest = "/how2compress/data/VisDrone-VAQ-MP4"
frames = 25  # Number of frames per chunk
qp = 30
aq_mode = 1

if not os.path.exists(dest):
    os.makedirs(dest)

for dataset in DATASETS:
    # Use the resized version of the sequence
    dataset_resized = f"{dataset}_resized"
    src_path = os.path.join(src_root, dataset_resized)
    
    if not os.path.exists(os.path.join(dest, dataset_resized)):
        os.makedirs(os.path.join(dest, dataset_resized))

    # Get all h264 files
    files = sorted([f for f in os.listdir(src_path) if f.endswith('.h264')])
    num = len(files)
    num_chunks = math.ceil(num / frames)

    for i in tqdm(range(num_chunks), desc=f"Converting {dataset_resized}"):
        # Calculate start and end frame numbers
        start_frame = i * frames + 1
        end_frame = min((i + 1) * frames, num)
        num_frames = end_frame - start_frame + 1

        # Construct input pattern for the chunk
        input_pattern = os.path.join(src_path, "%07d.h264")  # Using 7 digits for VisDrone

        subprocess.run(
            [
                "ffmpeg",
                "-i",
                input_pattern,
                "-start_number",
                str(start_frame),
                "-vframes",
                str(num_frames),
                "-framerate",
                "25",
                "-c:v",
                "libx264",
                "-qp",
                str(qp),
                "-x264-params",
                f"aq-mode={aq_mode}",
                os.path.join(dest, dataset_resized, f"{i+1:06d}.mp4"),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        ) 