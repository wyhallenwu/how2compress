import torch
import cv2
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model.hgmodel import MobileVitV2
from src.model.utils import ems2selections
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
from tabulate import tabulate

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model path")

args = parser.parse_args()

DEVICE = "cuda:0"
RESIZE_FACTOR = 4
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
BATCH_SIZE = 1

# Load YOLO model
inferencer = YOLO("yolov8m.pt", verbose=True).to(DEVICE)

# Setup paths
root = "data/visdrone/VisDrone2019-VID-val"
results_root = "/how2compress/results/visdrone_vaq"
if not os.path.exists(results_root):
    os.makedirs(results_root)

# Initialize dataset
dataset = VisDroneDataset(
    dataset_dir=root,
    reference_dir="/how2compress/data/VisDroneH264",
    yuv_dir="/how2compress/data/VisDroneYUV",
    resize_factor=RESIZE_FACTOR,
)

def decode_h264_frame(h264_path, frame_number):
    """Decode a specific frame from an H.264 file using ffmpeg."""
    # Create a temporary file for the decoded frame
    temp_file = f"/tmp/frame_{frame_number}.jpg"
    
    # Use ffmpeg to extract the specific frame using frame number
    cmd = [
        "ffmpeg",
        "-i", h264_path,
        "-vf", f"select=eq(n\\,{frame_number-1})",  # Select specific frame by number
        "-vframes", "1",  # Extract only one frame
        "-y",  # Overwrite output file if it exists
        "-f", "image2",  # Force image2 format
        "-q:v", "2",  # High quality
        "-pix_fmt", "rgb24",  # Use RGB format
        temp_file
    ]
    
    # Run ffmpeg command and capture output
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Check if ffmpeg command was successful
    if result.returncode != 0:
        print(f"Error decoding frame {frame_number} from {h264_path}")
        print(f"ffmpeg error: {result.stderr}")
        raise RuntimeError(f"Failed to decode frame {frame_number} from {h264_path}")
    
    # Check if the output file exists and has content
    if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
        print(f"Error: Temporary file {temp_file} was not created or is empty")
        print(f"ffmpeg output: {result.stdout}")
        print(f"ffmpeg error: {result.stderr}")
        raise RuntimeError(f"Failed to create temporary file for frame {frame_number}")
    
    # Read the decoded frame
    frame = cv2.imread(temp_file)
    if frame is None:
        print(f"Error: Could not read frame from {temp_file}")
        print(f"ffmpeg output: {result.stdout}")
        print(f"ffmpeg error: {result.stderr}")
        raise RuntimeError(f"Failed to read decoded frame {frame_number}")
    
    # Clean up temporary file
    try:
        os.remove(temp_file)
    except FileNotFoundError:
        print(f"Warning: Temporary file {temp_file} was already removed")
    
    # Resize frame to 1920x1088
    frame = cv2.resize(frame, (1920, 1088))
    
    # Process frame using the same pipeline as VisDroneDataset
    frame = image_ops.wrap_img(frame)  # Convert to RGB and normalize to [0,1]
    frame = dataset.transform_fn(frame)  # Apply YOLO transformation
    
    return frame

# Process each sequence
for seq in DATASETS:
    print(f"\nProcessing sequence: {seq}")
    
    # Setup sequence-specific paths
    r = os.path.join(results_root, f"eval_vaq1_vs_uniform_{seq}_resized.txt")
    enc_frames_dir = os.path.join(results_root, f"{seq}_resized")
    if not os.path.exists(enc_frames_dir):
        os.makedirs(enc_frames_dir)

    mAPs_vaq = []
    mAPs_uniform = []
    frames_size_vaq = []
    frames_size_uniform = []
    times = []

    # Load sequence into dataset
    dataset.load_sequence(seq)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False
    )
    
    for images, labels, indices in tqdm(dataloader, desc=f"{seq} val"):
        # Get ground truth detections from dataset images
        images = images.to(DEVICE)
        targets = inferencer.predict(images, classes=[0,2], verbose=False, save=False)
        targets = metrics.yolo2sv(targets)
        targets = [
            metrics.normalize_detections(
                det,
                (1920, 1088),
            )
            for det in targets
        ]
        
        # Get predictions on uniform AQ (QP 31) frames
        frame_idx = indices[0] + 1  # Convert to 1-based indexing
        uniform_path = os.path.join("/how2compress/data/VisDroneH264", f"{seq}_resized", "31", f"{frame_idx:07d}.h264")
        uniform_frame_tensor = decode_h264_frame(uniform_path, 1)  # Always get the first frame from each h264 file
        uniform_frame_tensor = uniform_frame_tensor.unsqueeze(0).to(DEVICE)
        preds_uniform = inferencer.predict(uniform_frame_tensor, classes=[0,2], verbose=False, save=False)
        preds_uniform = metrics.yolo2sv(preds_uniform)
        preds_uniform = [
            metrics.normalize_detections(
                det,
                (1920, 1088),
            )
            for det in preds_uniform
        ]
        
        # Get predictions on VAQ1 frames
        vaq_path = os.path.join("/how2compress/data/VisDroneH264VAQ1", f"{seq}_resized", f"{frame_idx:07d}.h264")
        vaq_frame_tensor = decode_h264_frame(vaq_path, 1)  # Always get the first frame from each h264 file
        vaq_frame_tensor = vaq_frame_tensor.unsqueeze(0).to(DEVICE)
        preds_vaq = inferencer.predict(vaq_frame_tensor, classes=[0,2], verbose=False, save=False)
        preds_vaq = metrics.yolo2sv(preds_vaq)
        preds_vaq = [
            metrics.normalize_detections(
                det,
                (1920, 1088),
            )
            for det in preds_vaq
        ]
        
        # Get frame sizes
        frame_size_uniform = os.path.getsize(uniform_path)
        frame_size_vaq = os.path.getsize(vaq_path)
        frames_size_uniform.append(frame_size_uniform)
        frames_size_vaq.append(frame_size_vaq)
        
        # Calculate mAP for both methods
        mAP_uniform = sv.MeanAveragePrecision.from_detections(preds_uniform, targets)
        mAP_vaq = sv.MeanAveragePrecision.from_detections(preds_vaq, targets)
        
        print(f"Frame {frame_idx}: Uniform mAP: {mAP_uniform.map50_95}, VAQ1 mAP: {mAP_vaq.map50_95}")
        mAPs_uniform.append(mAP_uniform)
        mAPs_vaq.append(mAP_vaq)

    # Save results for this sequence
    with open(r, "w") as f:
        f.write("frame,uniform_mAP50_95,uniform_mAP75,uniform_mAP50,vaq1_mAP50_95,vaq1_mAP75,vaq1_mAP50,uniform_size,vaq1_size\n")
        for i, (mAP_uniform, mAP_vaq, size_uniform, size_vaq) in enumerate(zip(mAPs_uniform, mAPs_vaq, frames_size_uniform, frames_size_vaq)):
            f.write(
                f"{i+1},{mAP_uniform.map50_95},{mAP_uniform.map75},{mAP_uniform.map50},"
                f"{mAP_vaq.map50_95},{mAP_vaq.map75},{mAP_vaq.map50},"
                f"{size_uniform},{size_vaq}\n"
            )
    
    # Calculate and print summary statistics
    avg_uniform_mAP50_95 = np.mean([m.map50_95 for m in mAPs_uniform])
    avg_uniform_mAP75 = np.mean([m.map75 for m in mAPs_uniform])
    avg_uniform_mAP50 = np.mean([m.map50 for m in mAPs_uniform])
    avg_vaq1_mAP50_95 = np.mean([m.map50_95 for m in mAPs_vaq])
    avg_vaq1_mAP75 = np.mean([m.map75 for m in mAPs_vaq])
    avg_vaq1_mAP50 = np.mean([m.map50 for m in mAPs_vaq])
    
    avg_uniform_size = np.mean(frames_size_uniform)
    avg_vaq1_size = np.mean(frames_size_vaq)
    
    summary_table = [
        ["Metric", "Uniform (QP31)", "VAQ1", "Improvement"],
        ["mAP@0.5:0.95", f"{avg_uniform_mAP50_95:.4f}", f"{avg_vaq1_mAP50_95:.4f}", f"{(avg_vaq1_mAP50_95 - avg_uniform_mAP50_95):.4f}"],
        ["mAP@0.75", f"{avg_uniform_mAP75:.4f}", f"{avg_vaq1_mAP75:.4f}", f"{(avg_vaq1_mAP75 - avg_uniform_mAP75):.4f}"],
        ["mAP@0.5", f"{avg_uniform_mAP50:.4f}", f"{avg_vaq1_mAP50:.4f}", f"{(avg_vaq1_mAP50 - avg_uniform_mAP50):.4f}"],
        ["Avg Frame Size (bytes)", f"{avg_uniform_size:.0f}", f"{avg_vaq1_size:.0f}", f"{(avg_vaq1_size - avg_uniform_size):.0f}"],
        ["Size Reduction (%)", "-", "-", f"{((avg_uniform_size - avg_vaq1_size) / avg_uniform_size * 100):.2f}%"]
    ]
    
    print(f"\nSummary for sequence {seq}:")
    print(tabulate(summary_table, headers="firstrow", tablefmt="grid"))
    
    # Save summary to a separate file
    summary_file = os.path.join(results_root, f"summary_{seq}_resized.txt")
    with open(summary_file, "w") as f:
        f.write(f"Summary for sequence {seq}:\n")
        f.write(tabulate(summary_table, headers="firstrow", tablefmt="grid")) 