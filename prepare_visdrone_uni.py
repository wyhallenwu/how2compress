import os
import subprocess
import math
import tempfile
import shutil
from tqdm import tqdm

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

# Setup paths and parameters
src_root = "/how2compress/data/visdrone/VisDrone2019-VID-val/sequences"
dest_qp25 = "/how2compress/data/VisDrone-UNI25-MP4"  # High quality QP25 videos
dest_qp30 = "/how2compress/data/VisDrone-UNI30-MP4"  # QP30 videos (uniform and AQ)
frames_per_chunk = 25  # Number of frames per chunk
qp25 = 25  # High quality QP value
qp30 = 30  # QP value for uniform and AQ
aq_mode = 1  # AQ mode for x264

# Create destination directories
for dest in [dest_qp25, dest_qp30]:
    if not os.path.exists(dest):
        os.makedirs(dest)

for dataset in DATASETS:
    # Use the resized version of the sequence
    dataset_resized = f"{dataset}_resized"
    src_path = os.path.join(src_root, dataset_resized)
    
    # Create sequence directories in both destinations
    for dest in [dest_qp25, dest_qp30]:
        if not os.path.exists(os.path.join(dest, dataset_resized)):
            os.makedirs(os.path.join(dest, dataset_resized))

    # Get all jpg files
    files = sorted([f for f in os.listdir(src_path) if f.endswith('.jpg')])
    num_frames = len(files)
    num_chunks = math.ceil(num_frames / frames_per_chunk)

    for i in tqdm(range(num_chunks), desc=f"Converting {dataset_resized}"):
        start_idx = i * frames_per_chunk + 1
        with tempfile.TemporaryDirectory() as tmpdirname:
            for j in range(frames_per_chunk):
                idx = start_idx + j
                if idx > num_frames:
                    break
                shutil.copyfile(
                    os.path.join(src_path, f"{idx:07d}.jpg"),
                    os.path.join(tmpdirname, f"{j:06d}.jpg"),
                )
            
            # First encode with QP25 for high quality
            subprocess.run(
                [
                    "/myh264/bin/ffmpeg",
                    "-y",
                    "-i",
                    f"{tmpdirname}/%06d.jpg",
                    "-start_number",
                    str(0),
                    "-vframes",
                    str(frames_per_chunk),
                    "-framerate",
                    str(frames_per_chunk),
                    "-qp",
                    str(qp25),
                    "-pix_fmt",
                    "yuv420p",
                    os.path.join(dest_qp25, dataset_resized, f"uni-{(i+1):02d}.mp4"),
                ]
            )

            # Then encode with QP30 for uniform quality
            subprocess.run(
                [
                    "/myh264/bin/ffmpeg",
                    "-y",
                    "-i",
                    f"{tmpdirname}/%06d.jpg",
                    "-start_number",
                    str(0),
                    "-vframes",
                    str(frames_per_chunk),
                    "-framerate",
                    str(frames_per_chunk),
                    "-qp",
                    str(qp30),
                    "-pix_fmt",
                    "yuv420p",
                    os.path.join(dest_qp30, dataset_resized, f"uni-{(i+1):02d}.mp4"),
                ]
            )

            # Finally encode with x264 with AQ mode 1 at QP30
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    os.path.join(dest_qp30, dataset_resized, f"uni-{(i+1):02d}.mp4"),
                    "-c:v",
                    "libx264",
                    "-qp",
                    str(qp30),
                    "-x264-params",
                    f"aq-mode={aq_mode}",
                    os.path.join(dest_qp30, dataset_resized, f"aq-{aq_mode}-{(i+1):02d}.mp4"),
                ]
            ) 