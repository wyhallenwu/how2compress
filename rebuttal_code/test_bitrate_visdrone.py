import torch
import cv2
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model.hgmodel import MobileVitV2
from src.model.utils import ems2selections
from src.model.am_model import AccMpeg
from src.utils import image_ops, load, cals, metrics
from src.dataset.dataloader import VisDroneDataset, collate_fn
import os
from ultralytics import YOLO
from tqdm import tqdm
import supervision as sv
import time
import argparse
import subprocess
import numpy as np
import tempfile
import shutil
from collections import defaultdict

# parser = argparse.ArgumentParser()
# parser.add_argument("--model", type=str, help="model path")
# parser.add_argument("--accmpeg_model", type=str, help="accmpeg model path")
# args = parser.parse_args()

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

# Mapping for QP values
mapping = {0: 45, 1: 43, 2: 37, 3: 34, 4: 30}

DEVICE = "cuda:0"
RESIZE_FACTOR = 4
BATCH_SIZE = 1
num = 25  # Number of frames per chunk
bf = 3    # Buffer frames

# Load models
model1 = AccMpeg(1088//16, 1920//16)  # Assuming 1920x1080 resolution
model1.load_state_dict(torch.load("pretrained/train/exp-accmpeg-1704-0.35q/1--0.007357695757508331.pth"))
model1.to(DEVICE)

model2 = MobileVitV2()
model2.load_state_dict(torch.load("pretrained/train/exp1704-n3/5-0.5452541371035616+-0.007466545299626204-0.993-0.965.pth"))
model2.to(DEVICE)
model2.set_output_size((1088//16, 1920//16))

# # Load YOLO model
# inferencer = YOLO("yolov8m.pt", verbose=True).to(DEVICE)

# Setup paths
root = "data/visdrone/VisDrone2019-VID-val"
results_root = "/how2compress/results/visdrone"
yuv_root = "/how2compress/data/VisDroneYUV25"
if not os.path.exists(results_root):
    os.makedirs(results_root)

# Initialize dataset
dataset = VisDroneDataset(
    dataset_dir=root,
    reference_dir="/how2compress/data/VisDroneH264",
    yuv_dir="/how2compress/data/VisDroneYUV",
    resize_factor=RESIZE_FACTOR,
)

transform = image_ops.vit_transform_fn()
resizer = image_ops.resize_img_tensor((1088//16 * 4, 1920//16 * 4))

# Dictionary to store statistics
all_stats = defaultdict(lambda: defaultdict(list))

# Process each sequence
for seq in DATASETS:
    print(f"\nProcessing sequence: {seq}")
    
    # Setup sequence-specific paths
    seq_dir = os.path.join(results_root, seq)
    if not os.path.exists(seq_dir):
        os.makedirs(seq_dir)

    # Get YUV directory path
    yuv_dir = os.path.join(yuv_root, f"{seq}_resized")

    dataset.load_sequence(seq)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False
    )

    # Process frames in chunks
    chunk_id = 0  # Sequential chunk ID
    for chunk_idx in tqdm(range(0, len(dataset), num), desc=f"Processing chunks for {seq}"):
        chunk_frames = []
        chunk_indices = []
        
        # Collect frames for this chunk
        for i in range(num):
            if chunk_idx + i >= len(dataset):
                break
            image, _, idx = dataset[chunk_idx + i]
            chunk_frames.append(image)
            chunk_indices.append(idx)
        
        # Skip this chunk if we don't have enough frames
        if len(chunk_frames) < num:
            continue

        # Create temporary directory for this chunk
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Save frames to temporary directory
            for i, frame in enumerate(chunk_frames):
                # Convert tensor to numpy array and adjust format for OpenCV
                frame_np = frame.cpu().numpy()  # Convert to numpy
                frame_np = (frame_np * 255).astype(np.uint8)  # Scale to [0,255]
                frame_np = np.transpose(frame_np, (1, 2, 0))  # Change from (C,H,W) to (H,W,C)
                # frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
                cv2.imwrite(os.path.join(tmpdirname, f"{i:07d}.jpg"), frame_np)

            # Process with AccMpeg model
            with open("/myh264/qp_matrix_file", "w") as f:
                for frame in chunk_frames:
                    # Convert tensor to numpy array and adjust format for OpenCV
                    frame_np = frame.cpu().numpy()  # Convert to numpy
                    frame_np = (frame_np * 255).astype(np.uint8)  # Scale to [0,255]
                    frame_np = np.transpose(frame_np, (1, 2, 0))  # Change from (C,H,W) to (H,W,C)
                    # frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
                    
                    # Now wrap the numpy array
                    image = image_ops.wrap_img(frame_np)
                    image = transform(image).unsqueeze(0).to(DEVICE)
                    resize_image = resizer(image)
                    ems_map_indices, ems_map_v, selections = model1(resize_image)
                    selections = [mapping[i] for _, i in selections[0]]
                    
                    matrix = np.reshape(selections, (1088//16, 1920//16))
                    for row in matrix:
                        f.write(" ".join(map(str, row)) + "\n")

            # Encode with AccMpeg
            accmpeg_output = os.path.join(seq_dir, f"accmpeg_chunk_{chunk_id:07d}.mp4")
            subprocess.run([
                "/myh264/bin/ffmpeg",
                "-y",
                "-i", f"{tmpdirname}/%07d.jpg",
                "-start_number", "0",
                "-vframes", str(len(chunk_frames)),
                "-framerate", "25",
                "-qp", "10",
                "-pix_fmt", "yuv420p",
                accmpeg_output
            ])

            # Process with our model
            with open("/myh264/qp_matrix_file", "w") as f:
                for frame in chunk_frames:
                    # Convert tensor to numpy array and adjust format for OpenCV
                    frame_np = frame.cpu().numpy()  # Convert to numpy
                    frame_np = (frame_np * 255).astype(np.uint8)  # Scale to [0,255]
                    frame_np = np.transpose(frame_np, (1, 2, 0))  # Change from (C,H,W) to (H,W,C)
                    frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
                    
                    # Now wrap the numpy array
                    image = image_ops.wrap_img(frame_np)
                    image = transform(image).unsqueeze(0).to(DEVICE)
                    resize_image = resizer(image)
                    ems_map_indices, ems_map_v, selections = model2(resize_image)
                    selections = [mapping[i] for _, i in selections[0]]
                    
                    matrix = np.reshape(selections, (1088//16, 1920//16))
                    for row in matrix:
                        f.write(" ".join(map(str, row)) + "\n")

            # Encode with our model
            ours_output = os.path.join(seq_dir, f"ours_chunk_{chunk_id:07d}.mp4")
            subprocess.run([
                "/myh264/bin/ffmpeg",
                "-y",
                "-i", f"{tmpdirname}/%07d.jpg",
                "-start_number", "0",
                "-vframes", str(len(chunk_frames)),
                "-framerate", "25",
                "-qp", "10",
                "-pix_fmt", "yuv420p",
                ours_output
            ])

            # Process with uniform QP using YUV file
            uniqp_output = os.path.join(seq_dir, f"uniqp_chunk_{chunk_id:07d}.h264")
            subprocess.run([
                "/how2compress/src/tools/AppEncCudaNoEM",
                "-i", os.path.join(yuv_dir, f"{chunk_id+1:07d}.yuv"),
                "-o", uniqp_output,
                "-s", "1920x1080",
                "-gpu", "0",
                "-qmin", "30",
                "-gop", str(len(chunk_frames)),
                "-qmax", "30",
                "-bf", str(bf),
                "-fps", str(len(chunk_frames)),
                "-constqp", "30",
                "-initqp", "30",
                "-tuninginfo", "ultralowlatency",
                "-rc", "constqp"
            ])

            # Process with spatial AQ using YUV file
            spatial_aq_output = os.path.join(seq_dir, f"spatial_aq_chunk_{chunk_id:07d}.h264")
            subprocess.run([
                "/how2compress/src/tools/AppEncCudaNoEM",
                "-i", os.path.join(yuv_dir, f"{chunk_id+1:07d}.yuv"),
                "-o", spatial_aq_output,
                "-s", "1920x1080",
                "-gop", str(len(chunk_frames)),
                "-qmin", "30",
                "-qmax", "45",
                "-bf", str(bf),
                "-fps", str(len(chunk_frames)),
                "-aq", "0",
                "-initqp", "35",
                "-gpu", "0",
                "-tuninginfo", "ultralowlatency",
                "-rc", "cbr"
            ])

            # Calculate and save bitrate statistics for this chunk
            chunk_stats = {}
            for method, output_file in [
                ("accmpeg", accmpeg_output),
                ("ours", ours_output),
                ("uniqp", uniqp_output),
                ("spatial_aq", spatial_aq_output)
            ]:
                file_size = os.path.getsize(output_file)
                bitrate = (file_size * 8 * 25) / (len(chunk_frames) * 1000000)  # Mbps
                chunk_stats[method] = bitrate
                # Store statistics for summary
                all_stats[seq][method].append(bitrate)
            
            # Save individual chunk results
            chunk_stats_file = os.path.join(seq_dir, f"chunk_{chunk_id:07d}_stats.txt")
            with open(chunk_stats_file, "w") as f:
                f.write(f"Chunk {chunk_id} Statistics\n")
                f.write("=====================\n\n")
                f.write(f"Frame Range: {chunk_indices[0]} to {chunk_indices[-1]}\n")
                f.write(f"Number of Frames: {len(chunk_frames)}\n\n")
                
                for method in ["accmpeg", "ours", "uniqp", "spatial_aq"]:
                    f.write(f"{method.upper()}:\n")
                    f.write(f"  Bitrate: {chunk_stats[method]:.2f} Mbps\n")
                    f.write(f"  Output File: {os.path.basename(output_file)}\n\n")
            
            chunk_id += 1  # Increment chunk ID for next chunk

    # Save sequence-level summary with average bitrates
    seq_summary_file = os.path.join(seq_dir, "sequence_summary.txt")
    with open(seq_summary_file, "w") as f:
        f.write(f"Sequence: {seq}\n")
        f.write("=" * (len(seq) + 11) + "\n\n")
        
        for method in ["accmpeg", "ours", "uniqp", "spatial_aq"]:
            bitrates = all_stats[seq][method]
            if bitrates:
                avg_bitrate = np.mean(bitrates)
                std_bitrate = np.std(bitrates)
                min_bitrate = np.min(bitrates)
                max_bitrate = np.max(bitrates)
                
                f.write(f"{method.upper()}:\n")
                f.write(f"  Average: {avg_bitrate:.2f} Mbps\n")
                f.write(f"  Std Dev: {std_bitrate:.2f} Mbps\n")
                f.write(f"  Min: {min_bitrate:.2f} Mbps\n")
                f.write(f"  Max: {max_bitrate:.2f} Mbps\n")
                f.write(f"  Number of Chunks: {len(bitrates)}\n\n")

# Write overall statistics
summary_file = os.path.join(results_root, "bitrate_summary.txt")
with open(summary_file, "w") as f:
    f.write("VisDrone Bitrate Statistics Summary\n")
    f.write("=================================\n\n")
    
    for seq in DATASETS:
        f.write(f"Sequence: {seq}\n")
        f.write("-" * (len(seq) + 11) + "\n")
        
        for method in ["accmpeg", "ours", "uniqp", "spatial_aq"]:
            bitrates = all_stats[seq][method]
            if bitrates:
                avg_bitrate = np.mean(bitrates)
                std_bitrate = np.std(bitrates)
                min_bitrate = np.min(bitrates)
                max_bitrate = np.max(bitrates)
                
                f.write(f"\n{method.upper()}:\n")
                f.write(f"  Average: {avg_bitrate:.2f} Mbps\n")
                f.write(f"  Std Dev: {std_bitrate:.2f} Mbps\n")
                f.write(f"  Min: {min_bitrate:.2f} Mbps\n")
                f.write(f"  Max: {max_bitrate:.2f} Mbps\n")
        
        f.write("\n" + "="*50 + "\n\n")
    
    # Write overall statistics
    f.write("Overall Statistics Across All Sequences\n")
    f.write("====================================\n\n")
    
    for method in ["accmpeg", "ours", "uniqp", "spatial_aq"]:
        all_bitrates = []
        for seq in DATASETS:
            all_bitrates.extend(all_stats[seq][method])
        
        if all_bitrates:
            avg_bitrate = np.mean(all_bitrates)
            std_bitrate = np.std(all_bitrates)
            min_bitrate = np.min(all_bitrates)
            max_bitrate = np.max(all_bitrates)
            
            f.write(f"\n{method.upper()}:\n")
            f.write(f"  Average: {avg_bitrate:.2f} Mbps\n")
            f.write(f"  Std Dev: {std_bitrate:.2f} Mbps\n")
            f.write(f"  Min: {min_bitrate:.2f} Mbps\n")
            f.write(f"  Max: {max_bitrate:.2f} Mbps\n") 