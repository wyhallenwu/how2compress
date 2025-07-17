import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import torch
from tqdm import tqdm
import os
import json
from datetime import datetime

def calculate_oks(gt_keypoints, pred_keypoints, bbox_area):
    """Calculate Object Keypoint Similarity (OKS) - standard COCO metric."""
    # COCO keypoint sigmas (17 keypoints for person)
    kpt_sigmas = np.array([
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
        0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
    ])
    
    # Calculate squared distances
    dx = gt_keypoints[:, 0] - pred_keypoints[:, 0]
    dy = gt_keypoints[:, 1] - pred_keypoints[:, 1]
    d_squared = dx**2 + dy**2
    
    # Only consider visible keypoints (confidence > threshold)
    visible = (gt_keypoints[:, 2] > 0.5) & (pred_keypoints[:, 2] > 0.5)
    
    if not np.any(visible):
        return 0.0
    
    # Calculate OKS
    s = np.sqrt(bbox_area)  # scale factor
    variances = (kpt_sigmas * 2) ** 2
    oks_per_kpt = np.exp(-d_squared / (2 * s**2 * variances))
    
    # Only average over visible keypoints
    oks = np.mean(oks_per_kpt[visible])
    return oks

def calculate_pck(gt_keypoints, pred_keypoints, threshold_ratio=0.2):
    """Calculate Percentage of Correct Keypoints (PCK)."""
    # Calculate head size for normalization (distance between ears or shoulders)
    head_keypoints = [3, 4]  # left_ear, right_ear
    if gt_keypoints[head_keypoints, 2].sum() > 1:  # both ears visible
        head_size = np.linalg.norm(gt_keypoints[3, :2] - gt_keypoints[4, :2])
    else:
        # Use shoulder distance as fallback
        shoulder_keypoints = [5, 6]  # left_shoulder, right_shoulder
        head_size = np.linalg.norm(gt_keypoints[5, :2] - gt_keypoints[6, :2])
    
    if head_size == 0:
        return 0.0
    
    threshold = threshold_ratio * head_size
    
    # Calculate distances
    distances = np.linalg.norm(gt_keypoints[:, :2] - pred_keypoints[:, :2], axis=1)
    
    # Only consider visible keypoints
    visible = (gt_keypoints[:, 2] > 0.5) & (pred_keypoints[:, 2] > 0.5)
    
    if not np.any(visible):
        return 0.0
    
    # Calculate PCK
    correct = distances[visible] < threshold
    pck = np.mean(correct)
    return pck

def calculate_mpjpe(gt_keypoints, pred_keypoints, normalize_by_bbox=True):
    """Calculate Mean Per Joint Position Error (MPJPE)."""
    # Only consider visible keypoints
    visible = (gt_keypoints[:, 2] > 0.5) & (pred_keypoints[:, 2] > 0.5)
    
    if not np.any(visible):
        return float('inf')
    
    # Calculate distances for visible keypoints
    distances = np.linalg.norm(gt_keypoints[visible, :2] - pred_keypoints[visible, :2], axis=1)
    
    # Normalize by bounding box size if requested
    if normalize_by_bbox:
        bbox_size = max(
            np.max(gt_keypoints[visible, 0]) - np.min(gt_keypoints[visible, 0]),
            np.max(gt_keypoints[visible, 1]) - np.min(gt_keypoints[visible, 1])
        )
        if bbox_size > 0:
            distances = distances / bbox_size
    
    return np.mean(distances)

def get_bbox_area(keypoints):
    """Calculate bounding box area from keypoints."""
    visible_kpts = keypoints[keypoints[:, 2] > 0.5]
    if len(visible_kpts) == 0:
        return 0
    
    x_min, y_min = np.min(visible_kpts[:, :2], axis=0)
    x_max, y_max = np.max(visible_kpts[:, :2], axis=0)
    return (x_max - x_min) * (y_max - y_min)

def match_keypoints(gt_keypoints, pred_keypoints):
    """Match keypoints between ground truth and prediction using Hungarian algorithm."""
    from scipy.optimize import linear_sum_assignment
    
    # Convert to numpy arrays if they're not already
    gt_kpts = gt_keypoints.cpu().numpy() if torch.is_tensor(gt_keypoints) else np.array(gt_keypoints)
    pred_kpts = pred_keypoints.cpu().numpy() if torch.is_tensor(pred_keypoints) else np.array(pred_keypoints)
    
    # Check if arrays are empty or have wrong shape
    if len(gt_kpts.shape) != 3 or len(pred_kpts.shape) != 3:
        return None, None
    
    if gt_kpts.shape[0] == 0 or pred_kpts.shape[0] == 0:
        return None, None
    
    # Ensure we have the right shape (num_persons, num_keypoints, 3)
    if gt_kpts.shape[1] != 17 or pred_kpts.shape[1] != 17:
        return None, None
    
    # Calculate cost matrix using OKS-based distance
    n_gt = len(gt_kpts)
    n_pred = len(pred_kpts)
    cost_matrix = np.zeros((n_gt, n_pred))
    
    for i in range(n_gt):
        for j in range(n_pred):
            # Use OKS as similarity measure (convert to cost)
            bbox_area = get_bbox_area(gt_kpts[i])
            if bbox_area > 0:
                oks = calculate_oks(gt_kpts[i], pred_kpts[j], bbox_area)
                cost_matrix[i, j] = 1 - oks  # Convert similarity to cost
            else:
                cost_matrix[i, j] = 1.0  # Maximum cost if no valid bbox
    
    # Use Hungarian algorithm to find optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Return matched pairs
    matched_gt = gt_kpts[row_ind]
    matched_pred = pred_kpts[col_ind]
    
    return matched_gt, matched_pred

def calculate_pose_metrics(gt_keypoints, pred_keypoints):
    """Calculate comprehensive pose estimation metrics."""
    # Match keypoints between GT and prediction
    matched_gt, matched_pred = match_keypoints(gt_keypoints, pred_keypoints)
    
    # If matching failed, return None
    if matched_gt is None or matched_pred is None:
        return None
    
    metrics = {}
    person_metrics = []
    
    # Calculate metrics for each matched person
    for i in range(len(matched_gt)):
        gt_person = matched_gt[i]
        pred_person = matched_pred[i]
        
        # Calculate bbox area for OKS
        bbox_area = get_bbox_area(gt_person)
        
        person_result = {
            'oks': calculate_oks(gt_person, pred_person, bbox_area),
            'pck_0.2': calculate_pck(gt_person, pred_person, threshold_ratio=0.2),
            'pck_0.5': calculate_pck(gt_person, pred_person, threshold_ratio=0.5),
            'mpjpe_normalized': calculate_mpjpe(gt_person, pred_person, normalize_by_bbox=True),
            'mpjpe_pixels': calculate_mpjpe(gt_person, pred_person, normalize_by_bbox=False),
            'visible_keypoints': np.sum((gt_person[:, 2] > 0.5) & (pred_person[:, 2] > 0.5))
        }
        person_metrics.append(person_result)
    
    if not person_metrics:
        return None
    
    # Average metrics across all persons
    metrics = {
        'oks': np.mean([p['oks'] for p in person_metrics]),
        'pck_0.2': np.mean([p['pck_0.2'] for p in person_metrics]),
        'pck_0.5': np.mean([p['pck_0.5'] for p in person_metrics]),
        'mpjpe_normalized': np.mean([p['mpjpe_normalized'] for p in person_metrics]),
        'mpjpe_pixels': np.mean([p['mpjpe_pixels'] for p in person_metrics]),
        'avg_visible_keypoints': np.mean([p['visible_keypoints'] for p in person_metrics]),
        'num_matched': len(matched_gt)
    }
    
    return metrics

def load_video(video_path):
    """Load video and return frames."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def analyze_video_chunks(high_quality_chunks, low_quality_chunks, model):
    """Analyze a pair of video chunk lists."""
    all_metrics = []
    
    # Process each chunk pair
    for gt_chunk, pred_chunk in tqdm(zip(high_quality_chunks, low_quality_chunks), total=len(high_quality_chunks)):
        # Load videos
        gt_frames = load_video(gt_chunk)
        pred_frames = load_video(pred_chunk)
        
        if len(gt_frames) != len(pred_frames):
            print(f"Warning: Frame count mismatch in chunks. GT: {len(gt_frames)}, Pred: {len(pred_frames)}")
            min_frames = min(len(gt_frames), len(pred_frames))
            gt_frames = gt_frames[:min_frames]
            pred_frames = pred_frames[:min_frames]
        
        # Process each frame pair
        for gt_frame, pred_frame in zip(gt_frames, pred_frames):
            # Get predictions
            gt_results = model(gt_frame, verbose=False)
            pred_results = model(pred_frame, verbose=False)
            
            # Extract keypoints
            gt_keypoints = gt_results[0].keypoints.data if gt_results[0].keypoints is not None else torch.empty(0, 17, 3)
            pred_keypoints = pred_results[0].keypoints.data if pred_results[0].keypoints is not None else torch.empty(0, 17, 3)
            
            # Skip if no detections in either frame
            if len(gt_keypoints) == 0 or len(pred_keypoints) == 0:
                continue
            
            # Calculate metrics
            metrics = calculate_pose_metrics(gt_keypoints, pred_keypoints)
            if metrics is not None:
                all_metrics.append(metrics)
    
    if not all_metrics:
        print("Warning: No valid frames with detections found!")
        return {
            'oks': float('nan'),
            'pck_0.2': float('nan'),
            'pck_0.5': float('nan'),
            'mpjpe_normalized': float('nan'),
            'mpjpe_pixels': float('nan'),
            'avg_visible_keypoints': 0,
            'num_matched': 0
        }
    
    # Calculate average metrics across all frames
    avg_metrics = {
        'oks': np.mean([m['oks'] for m in all_metrics]),
        'pck_0.2': np.mean([m['pck_0.2'] for m in all_metrics]),
        'pck_0.5': np.mean([m['pck_0.5'] for m in all_metrics]),
        'mpjpe_normalized': np.mean([m['mpjpe_normalized'] for m in all_metrics]),
        'mpjpe_pixels': np.mean([m['mpjpe_pixels'] for m in all_metrics]),
        'avg_visible_keypoints': np.mean([m['avg_visible_keypoints'] for m in all_metrics]),
        'avg_matched_persons': np.mean([m['num_matched'] for m in all_metrics])
    }
    
    return avg_metrics

def write_results_to_file(results, sequences, output_file):
    """Write results to a file in a well-organized format."""
    with open(output_file, 'w') as f:
        # Write header with timestamp
        f.write(f"VisDrone Pose Estimation Quality Analysis Results\n")
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
                    f.write(f"  OKS: {metrics['oks']:.4f}\n")
                    f.write(f"  PCK@0.2: {metrics['pck_0.2']:.4f}\n")
                    f.write(f"  PCK@0.5: {metrics['pck_0.5']:.4f}\n")
                    f.write(f"  MPJPE (normalized): {metrics['mpjpe_normalized']:.4f}\n")
                    f.write(f"  MPJPE (pixels): {metrics['mpjpe_pixels']:.2f}\n")
                    f.write(f"  Average Visible Keypoints: {metrics['avg_visible_keypoints']:.1f}\n")
                    f.write(f"  Average Matched Persons: {metrics['avg_matched_persons']:.2f}\n")
            
            # Write comparison for this sequence
            if seq_name in results['ours'] and seq_name in results['accmpeg']:
                f.write("\nComparison:\n")
                f.write(f"  OKS: Ours {results['ours'][seq_name]['oks']:.4f} vs Accmpeg {results['accmpeg'][seq_name]['oks']:.4f}\n")
                f.write(f"  PCK@0.2: Ours {results['ours'][seq_name]['pck_0.2']:.4f} vs Accmpeg {results['accmpeg'][seq_name]['pck_0.2']:.4f}\n")
                f.write(f"  PCK@0.5: Ours {results['ours'][seq_name]['pck_0.5']:.4f} vs Accmpeg {results['accmpeg'][seq_name]['pck_0.5']:.4f}\n")
                f.write(f"  MPJPE (normalized): Ours {results['ours'][seq_name]['mpjpe_normalized']:.4f} vs Accmpeg {results['accmpeg'][seq_name]['mpjpe_normalized']:.4f}\n")
                f.write(f"  MPJPE (pixels): Ours {results['ours'][seq_name]['mpjpe_pixels']:.2f} vs Accmpeg {results['accmpeg'][seq_name]['mpjpe_pixels']:.2f}\n")
            
            f.write("\n" + "="*80 + "\n\n")
        
        # Write overall summary
        f.write("OVERALL SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        for method in ['ours', 'accmpeg', 'uniqp', 'spatial_aq']:
            f.write(f"{method.upper()} METHOD:\n")
            f.write("-"*40 + "\n")
            
            # Calculate averages for each metric
            all_oks = [r['oks'] for r in results[method].values() if not np.isnan(r['oks'])]
            all_pck_02 = [r['pck_0.2'] for r in results[method].values() if not np.isnan(r['pck_0.2'])]
            all_pck_05 = [r['pck_0.5'] for r in results[method].values() if not np.isnan(r['pck_0.5'])]
            all_mpjpe_norm = [r['mpjpe_normalized'] for r in results[method].values() if not np.isnan(r['mpjpe_normalized'])]
            all_mpjpe_pixels = [r['mpjpe_pixels'] for r in results[method].values() if not np.isnan(r['mpjpe_pixels'])]
            
            if all_oks:
                f.write(f"Average OKS: {np.mean(all_oks):.4f} ± {np.std(all_oks):.4f}\n")
                f.write(f"Average PCK@0.2: {np.mean(all_pck_02):.4f} ± {np.std(all_pck_02):.4f}\n")
                f.write(f"Average PCK@0.5: {np.mean(all_pck_05):.4f} ± {np.std(all_pck_05):.4f}\n")
                f.write(f"Average MPJPE (normalized): {np.mean(all_mpjpe_norm):.4f} ± {np.std(all_mpjpe_norm):.4f}\n")
                f.write(f"Average MPJPE (pixels): {np.mean(all_mpjpe_pixels):.2f} ± {np.std(all_mpjpe_pixels):.2f}\n")
            
            f.write("\n")

def main():
    # Initialize YOLO model
    model = YOLO("yolo11m-pose.pt")
    
    # Define paths
    high_quality_base = Path("/how2compress/data/VisDrone-UNI25-MP4")
    low_quality_base = Path("/how2compress/results/visdrone")
    
    # Define sequences to process
    sequences = [
        "uav0000086_00000_v",
        # "uav0000117_02622_v",
        # "uav0000137_00458_v",
        # "uav0000182_00000_v",
        # "uav0000268_05773_v",
        # "uav0000305_00000_v",
        # "uav0000339_00001_v"
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
        
        # Find the high quality video chunks
        high_quality_chunks = sorted(list(high_quality_base.glob(f"{seq_name}_resized/uni*.mp4")))
        if not high_quality_chunks:
            print(f"No high quality video chunks found for {seq_name}")
            continue
        
        print(f"Found {len(high_quality_chunks)} high quality chunks")
        
        # Find corresponding low quality video chunks
        seq_folder = low_quality_base / seq_name
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
            metrics = analyze_video_chunks(high_quality_chunks, ours_chunks, model)
            results['ours'][seq_name] = metrics
            
            print(f"Results for {seq_name} (ours):")
            print(f"OKS (Object Keypoint Similarity): {metrics['oks']:.4f}")
            print(f"PCK@0.2 (head-normalized): {metrics['pck_0.2']:.4f}")
            print(f"PCK@0.5 (head-normalized): {metrics['pck_0.5']:.4f}")
            print(f"MPJPE (normalized): {metrics['mpjpe_normalized']:.4f}")
            print(f"MPJPE (pixels): {metrics['mpjpe_pixels']:.2f}")
            print(f"Average Visible Keypoints: {metrics['avg_visible_keypoints']:.1f}")
            print(f"Average Matched Persons: {metrics['avg_matched_persons']:.2f}")
        
        # Get accmpeg chunks
        accmpeg_chunks = sorted(list(seq_folder.glob("accmpeg_chunk_*.mp4")))
        if not accmpeg_chunks:
            print(f"Warning: No 'accmpeg' chunks found for {seq_name}")
        else:
            print(f"Found {len(accmpeg_chunks)} 'accmpeg' chunks")
            
            # Process accmpeg comparison
            print(f"\nComparing high quality vs 'accmpeg' for {seq_name}")
            metrics = analyze_video_chunks(high_quality_chunks, accmpeg_chunks, model)
            results['accmpeg'][seq_name] = metrics
            
            print(f"Results for {seq_name} (accmpeg):")
            print(f"OKS (Object Keypoint Similarity): {metrics['oks']:.4f}")
            print(f"PCK@0.2 (head-normalized): {metrics['pck_0.2']:.4f}")
            print(f"PCK@0.5 (head-normalized): {metrics['pck_0.5']:.4f}")
            print(f"MPJPE (normalized): {metrics['mpjpe_normalized']:.4f}")
            print(f"MPJPE (pixels): {metrics['mpjpe_pixels']:.2f}")
            print(f"Average Visible Keypoints: {metrics['avg_visible_keypoints']:.1f}")
            print(f"Average Matched Persons: {metrics['avg_matched_persons']:.2f}")
        
        # Get uniform QP chunks
        uniqp_chunks = sorted(list(seq_folder.glob("uniqp_chunk_*.h264")))
        if not uniqp_chunks:
            print(f"Warning: No 'uniqp' chunks found for {seq_name}")
        else:
            print(f"Found {len(uniqp_chunks)} 'uniqp' chunks")
            
            # Process uniform QP comparison
            print(f"\nComparing high quality vs 'uniqp' for {seq_name}")
            metrics = analyze_video_chunks(high_quality_chunks, uniqp_chunks, model)
            results['uniqp'][seq_name] = metrics
            
            print(f"Results for {seq_name} (uniqp):")
            print(f"OKS (Object Keypoint Similarity): {metrics['oks']:.4f}")
            print(f"PCK@0.2 (head-normalized): {metrics['pck_0.2']:.4f}")
            print(f"PCK@0.5 (head-normalized): {metrics['pck_0.5']:.4f}")
            print(f"MPJPE (normalized): {metrics['mpjpe_normalized']:.4f}")
            print(f"MPJPE (pixels): {metrics['mpjpe_pixels']:.2f}")
            print(f"Average Visible Keypoints: {metrics['avg_visible_keypoints']:.1f}")
            print(f"Average Matched Persons: {metrics['avg_matched_persons']:.2f}")
        
        # Get spatial AQ chunks
        spatial_aq_chunks = sorted(list(seq_folder.glob("spatial_aq_chunk_*.h264")))
        if not spatial_aq_chunks:
            print(f"Warning: No 'spatial_aq' chunks found for {seq_name}")
        else:
            print(f"Found {len(spatial_aq_chunks)} 'spatial_aq' chunks")
            
            # Process spatial AQ comparison
            print(f"\nComparing high quality vs 'spatial_aq' for {seq_name}")
            metrics = analyze_video_chunks(high_quality_chunks, spatial_aq_chunks, model)
            results['spatial_aq'][seq_name] = metrics
            
            print(f"Results for {seq_name} (spatial_aq):")
            print(f"OKS (Object Keypoint Similarity): {metrics['oks']:.4f}")
            print(f"PCK@0.2 (head-normalized): {metrics['pck_0.2']:.4f}")
            print(f"PCK@0.5 (head-normalized): {metrics['pck_0.5']:.4f}")
            print(f"MPJPE (normalized): {metrics['mpjpe_normalized']:.4f}")
            print(f"MPJPE (pixels): {metrics['mpjpe_pixels']:.2f}")
            print(f"Average Visible Keypoints: {metrics['avg_visible_keypoints']:.1f}")
            print(f"Average Matched Persons: {metrics['avg_matched_persons']:.2f}")
    
    # Write results to file
    output_file = f"visdrone_pose_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    write_results_to_file(results, sequences, output_file)
    print(f"\nResults have been written to: {output_file}")

if __name__ == "__main__":
    main() 