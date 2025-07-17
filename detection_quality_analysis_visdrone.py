import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import torch
from tqdm import tqdm
import os
from datetime import datetime
import supervision as sv
from tabulate import tabulate
import subprocess
from src.utils import image_ops, metrics
from src.dataset.dataloader import VisDroneDataset, collate_fn
from torch.utils.data import DataLoader

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes."""
    # Convert to [x1, y1, x2, y2] format if needed
    if len(box1) == 4 and len(box2) == 4:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0
    return 0

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

def analyze_detection_quality(gt_frames, pred_frames, model, target_classes=[0, 2]):
    """Analyze detection quality between ground truth and prediction frames."""
    all_metrics = []
    
    # Process each frame pair
    for gt_frame, pred_frame in tqdm(zip(gt_frames, pred_frames), total=len(gt_frames)):
        # Get predictions
        gt_results = model(gt_frame, verbose=False)
        pred_results = model(pred_frame, verbose=False)
        
        # Extract boxes and convert to supervision format
        gt_detections = metrics.yolo2sv(gt_results)
        pred_detections = metrics.yolo2sv(pred_results)
        
        # Normalize detections
        gt_detections = [
            metrics.normalize_detections(det, (1920, 1088))
            for det in gt_detections
        ]
        pred_detections = [
            metrics.normalize_detections(det, (1920, 1088))
            for det in pred_detections
        ]
        
        # Calculate mAP metrics
        mAP = sv.MeanAveragePrecision.from_detections(pred_detections, gt_detections)
        
        metrics_dict = {
            'map50_95': mAP.map50_95,
            'map75': mAP.map75,
            'map50': mAP.map50,
            'num_gt': len(gt_detections[0]) if len(gt_detections) > 0 else 0,
            'num_pred': len(pred_detections[0]) if len(pred_detections) > 0 else 0
        }
        
        all_metrics.append(metrics_dict)
    
    if not all_metrics:
        print("Warning: No valid frames with detections found!")
        return {
            'map50_95': float('nan'),
            'map75': float('nan'),
            'map50': float('nan'),
            'num_gt': 0,
            'num_pred': 0
        }
    
    # Calculate average metrics across all frames
    avg_metrics = {
        'map50_95': np.mean([m['map50_95'] for m in all_metrics]),
        'map75': np.mean([m['map75'] for m in all_metrics]),
        'map50': np.mean([m['map50'] for m in all_metrics]),
        'num_gt': np.sum([m['num_gt'] for m in all_metrics]),
        'num_pred': np.sum([m['num_pred'] for m in all_metrics])
    }
    
    return avg_metrics

def write_results_to_file(results, sequences, output_file):
    """Write results to a file in a well-organized format."""
    with open(output_file, 'w') as f:
        # Write header with timestamp
        f.write(f"Detection Quality Analysis Results\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Write per-sequence results
        f.write("PER-SEQUENCE RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for seq_name in sequences:
            f.write(f"Sequence: {seq_name}\n")
            f.write("-"*40 + "\n")
            
            for method in ['ours', 'accmpeg', 'uniqp', 'spatial_aq']:
                if seq_name in results[method]:
                    metrics = results[method][seq_name]
                    f.write(f"\n{method.upper()} Method:\n")
                    f.write(f"  mAP@0.5:0.95: {metrics['map50_95']:.4f}\n")
                    f.write(f"  mAP@0.75: {metrics['map75']:.4f}\n")
                    f.write(f"  mAP@0.5: {metrics['map50']:.4f}\n")
                    f.write(f"  Total Ground Truth Objects: {metrics['num_gt']}\n")
                    f.write(f"  Total Predicted Objects: {metrics['num_pred']}\n")
            
            # Write comparison for this sequence
            if seq_name in results['ours'] and seq_name in results['accmpeg']:
                f.write("\nComparison:\n")
                f.write(f"  mAP@0.5:0.95: Ours {results['ours'][seq_name]['map50_95']:.4f} vs Accmpeg {results['accmpeg'][seq_name]['map50_95']:.4f}\n")
                f.write(f"  mAP@0.75: Ours {results['ours'][seq_name]['map75']:.4f} vs Accmpeg {results['accmpeg'][seq_name]['map75']:.4f}\n")
                f.write(f"  mAP@0.5: Ours {results['ours'][seq_name]['map50']:.4f} vs Accmpeg {results['accmpeg'][seq_name]['map50']:.4f}\n")
            
            if seq_name in results['ours'] and seq_name in results['uniqp']:
                f.write("\nComparison:\n")
                f.write(f"  mAP@0.5:0.95: Ours {results['ours'][seq_name]['map50_95']:.4f} vs Uniqp {results['uniqp'][seq_name]['map50_95']:.4f}\n")
                f.write(f"  mAP@0.75: Ours {results['ours'][seq_name]['map75']:.4f} vs Uniqp {results['uniqp'][seq_name]['map75']:.4f}\n")
                f.write(f"  mAP@0.5: Ours {results['ours'][seq_name]['map50']:.4f} vs Uniqp {results['uniqp'][seq_name]['map50']:.4f}\n")
            
            if seq_name in results['ours'] and seq_name in results['spatial_aq']:
                f.write("\nComparison:\n")
                f.write(f"  mAP@0.5:0.95: Ours {results['ours'][seq_name]['map50_95']:.4f} vs Spatial AQ {results['spatial_aq'][seq_name]['map50_95']:.4f}\n")
                f.write(f"  mAP@0.75: Ours {results['ours'][seq_name]['map75']:.4f} vs Spatial AQ {results['spatial_aq'][seq_name]['map75']:.4f}\n")
                f.write(f"  mAP@0.5: Ours {results['ours'][seq_name]['map50']:.4f} vs Spatial AQ {results['spatial_aq'][seq_name]['map50']:.4f}\n")
            
            f.write("\n" + "="*80 + "\n\n")
        
        # Write overall summary
        f.write("OVERALL SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        for method in ['ours', 'accmpeg', 'uniqp', 'spatial_aq']:
            f.write(f"{method.upper()} METHOD:\n")
            f.write("-"*40 + "\n")
            
            # Calculate averages for each metric
            all_map50_95 = [r['map50_95'] for r in results[method].values() if not np.isnan(r['map50_95'])]
            all_map75 = [r['map75'] for r in results[method].values() if not np.isnan(r['map75'])]
            all_map50 = [r['map50'] for r in results[method].values() if not np.isnan(r['map50'])]
            
            if all_map50_95:
                f.write(f"Average mAP@0.5:0.95: {np.mean(all_map50_95):.4f} ± {np.std(all_map50_95):.4f}\n")
                f.write(f"Average mAP@0.75: {np.mean(all_map75):.4f} ± {np.std(all_map75):.4f}\n")
                f.write(f"Average mAP@0.5: {np.mean(all_map50):.4f} ± {np.std(all_map50):.4f}\n")
            
            f.write("\n")

def main():
    # Initialize YOLO model
    model = YOLO("yolo11x.pt").to("cuda:0")
    
    # Define paths
    high_quality_base = Path("/how2compress/data/VisDrone-UNI25-MP4")  # QP25 videos as high quality
    low_quality_base = Path("/how2compress/data/VisDrone-UNI30-MP4")   # QP30 videos for uniform QP and spatial AQ
    results_base = Path("/how2compress/results/visdrone")              # Results for our method and AccMPEG
    
    # Define sequences to process
    sequences = [
        "uav0000086_00000_v",
        "uav0000117_02622_v",
        "uav0000137_00458_v",
        "uav0000182_00000_v",
        "uav0000268_05773_v",
        "uav0000305_00000_v",
        "uav0000339_00001_v"
    ]
    
    # Process each sequence
    results = {
        'ours': {},
        'accmpeg': {},
        'uniqp': {},
        'spatial_aq': {}
    }
    
    for seq_name in sequences:
        print(f"\nProcessing sequence: {seq_name}")
        
        # Find the high quality video chunks (QP25)
        high_quality_chunks = sorted(list(high_quality_base.glob(f"{seq_name}_resized/uni*.mp4")))
        if not high_quality_chunks:
            print(f"No high quality video chunks found for {seq_name}")
            continue
        
        print(f"Found {len(high_quality_chunks)} high quality chunks")
        
        # Find corresponding low quality video chunks
        seq_folder = results_base / seq_name
        if not seq_folder.exists():
            print(f"Warning: Sequence folder not found: {seq_folder}")
            continue
        
        # Get our method chunks
        ours_chunks = sorted(list(seq_folder.glob("ours_chunk_*.mp4")))
        if not ours_chunks:
            print(f"Warning: No 'ours' chunks found for {seq_name}")
        else:
            print(f"Found {len(ours_chunks)} 'ours' chunks")
            
            # Process ours comparison
            print(f"\nComparing high quality vs 'ours' for {seq_name}")
            metrics = analyze_detection_quality(high_quality_chunks, ours_chunks, model)
            results['ours'][seq_name] = metrics
            
            print(f"Results for {seq_name} (ours):")
            print(f"mAP@0.5:0.95: {metrics['map50_95']:.4f}")
            print(f"mAP@0.75: {metrics['map75']:.4f}")
            print(f"mAP@0.5: {metrics['map50']:.4f}")
        
        # Get accmpeg chunks
        accmpeg_chunks = sorted(list(seq_folder.glob("accmpeg_chunk_*.mp4")))
        if not accmpeg_chunks:
            print(f"Warning: No 'accmpeg' chunks found for {seq_name}")
        else:
            print(f"Found {len(accmpeg_chunks)} 'accmpeg' chunks")
            
            # Process accmpeg comparison
            print(f"\nComparing high quality vs 'accmpeg' for {seq_name}")
            metrics = analyze_detection_quality(high_quality_chunks, accmpeg_chunks, model)
            results['accmpeg'][seq_name] = metrics
            
            print(f"Results for {seq_name} (accmpeg):")
            print(f"mAP@0.5:0.95: {metrics['map50_95']:.4f}")
            print(f"mAP@0.75: {metrics['map75']:.4f}")
            print(f"mAP@0.5: {metrics['map50']:.4f}")
        
        # Get uniform QP chunks (QP30)
        uniqp_chunks = sorted(list(low_quality_base.glob(f"{seq_name}_resized/uni*.mp4")))
        if not uniqp_chunks:
            print(f"Warning: No 'uniqp' chunks found for {seq_name}")
        else:
            print(f"Found {len(uniqp_chunks)} 'uniqp' chunks")
            
            # Process uniform QP comparison
            print(f"\nComparing high quality vs 'uniqp' for {seq_name}")
            metrics = analyze_detection_quality(high_quality_chunks, uniqp_chunks, model)
            results['uniqp'][seq_name] = metrics
            
            print(f"Results for {seq_name} (uniqp):")
            print(f"mAP@0.5:0.95: {metrics['map50_95']:.4f}")
            print(f"mAP@0.75: {metrics['map75']:.4f}")
            print(f"mAP@0.5: {metrics['map50']:.4f}")
        
        # Get spatial AQ chunks (QP30 with AQ mode 1)
        spatial_aq_chunks = sorted(list(low_quality_base.glob(f"{seq_name}_resized/aq-1-*.mp4")))
        if not spatial_aq_chunks:
            print(f"Warning: No 'spatial_aq' chunks found for {seq_name}")
        else:
            print(f"Found {len(spatial_aq_chunks)} 'spatial_aq' chunks")
            
            # Process spatial AQ comparison
            print(f"\nComparing high quality vs 'spatial_aq' for {seq_name}")
            metrics = analyze_detection_quality(high_quality_chunks, spatial_aq_chunks, model)
            results['spatial_aq'][seq_name] = metrics
            
            print(f"Results for {seq_name} (spatial_aq):")
            print(f"mAP@0.5:0.95: {metrics['map50_95']:.4f}")
            print(f"mAP@0.75: {metrics['map75']:.4f}")
            print(f"mAP@0.5: {metrics['map50']:.4f}")
    
    # Write results to file
    output_file = f"visdrone_detection_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    write_results_to_file(results, sequences, output_file)
    print(f"\nResults have been written to: {output_file}")

if __name__ == "__main__":
    main() 