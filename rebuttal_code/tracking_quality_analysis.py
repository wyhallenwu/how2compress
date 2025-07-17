import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import torch
from tqdm import tqdm
import os
from datetime import datetime

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

def match_detections(gt_boxes, pred_boxes, iou_threshold=0.5):
    """Match detections between ground truth and predictions using Hungarian algorithm."""
    from scipy.optimize import linear_sum_assignment
    
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return [], []
    
    # Calculate cost matrix (1 - IoU)
    cost_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            cost_matrix[i, j] = 1 - calculate_iou(gt_box, pred_box)
    
    # Use Hungarian algorithm to find optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Filter matches by IoU threshold
    matches = []
    unmatched_gt = []
    unmatched_pred = list(range(len(pred_boxes)))
    
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < (1 - iou_threshold):
            matches.append((i, j))
            unmatched_pred.remove(j)
        else:
            unmatched_gt.append(i)
    
    return matches, unmatched_gt, unmatched_pred

def calculate_tracking_metrics(gt_boxes, pred_boxes, gt_ids, pred_ids):
    """Calculate tracking metrics including MOTA, MOTP, and ID metrics."""
    matches, unmatched_gt, unmatched_pred = match_detections(gt_boxes, pred_boxes)
    
    # Initialize metrics
    metrics = {
        'num_gt': len(gt_boxes),
        'num_pred': len(pred_boxes),
        'num_matches': len(matches),
        'num_false_positives': len(unmatched_pred),
        'num_false_negatives': len(unmatched_gt),
        'id_switches': 0,
        'mota': 0.0,
        'motp': 0.0,
        'idf1': 0.0
    }
    
    if len(matches) > 0:
        # Calculate MOTP (average IoU of matched detections)
        ious = []
        for gt_idx, pred_idx in matches:
            iou = calculate_iou(gt_boxes[gt_idx], pred_boxes[pred_idx])
            ious.append(iou)
        metrics['motp'] = np.mean(ious)
        
        # Calculate ID switches
        for gt_idx, pred_idx in matches:
            if gt_ids[gt_idx] != pred_ids[pred_idx]:
                metrics['id_switches'] += 1
        
        # Calculate MOTA
        metrics['mota'] = 1 - (metrics['num_false_positives'] + 
                             metrics['num_false_negatives'] + 
                             metrics['id_switches']) / max(metrics['num_gt'], 1)
        
        # Calculate IDF1
        num_correct_id = len(matches) - metrics['id_switches']
        metrics['idf1'] = 2 * num_correct_id / (metrics['num_gt'] + metrics['num_pred'])
    
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
            
            # Extract boxes and track IDs
            gt_boxes = gt_results[0].boxes.xyxy.cpu().numpy() if gt_results[0].boxes is not None else np.empty((0, 4))
            pred_boxes = pred_results[0].boxes.xyxy.cpu().numpy() if pred_results[0].boxes is not None else np.empty((0, 4))
            
            gt_ids = gt_results[0].boxes.id.cpu().numpy() if gt_results[0].boxes.id is not None else np.zeros(len(gt_boxes))
            pred_ids = pred_results[0].boxes.id.cpu().numpy() if pred_results[0].boxes.id is not None else np.zeros(len(pred_boxes))
            
            # Skip if no detections in either frame
            if len(gt_boxes) == 0 or len(pred_boxes) == 0:
                continue
            
            # Calculate metrics
            metrics = calculate_tracking_metrics(gt_boxes, pred_boxes, gt_ids, pred_ids)
            all_metrics.append(metrics)
    
    if not all_metrics:
        print("Warning: No valid frames with detections found!")
        return {
            'mota': float('nan'),
            'motp': float('nan'),
            'idf1': float('nan'),
            'num_gt': 0,
            'num_pred': 0,
            'num_matches': 0,
            'num_false_positives': 0,
            'num_false_negatives': 0,
            'id_switches': 0
        }
    
    # Calculate average metrics across all frames
    avg_metrics = {
        'mota': np.mean([m['mota'] for m in all_metrics]),
        'motp': np.mean([m['motp'] for m in all_metrics]),
        'idf1': np.mean([m['idf1'] for m in all_metrics]),
        'num_gt': np.sum([m['num_gt'] for m in all_metrics]),
        'num_pred': np.sum([m['num_pred'] for m in all_metrics]),
        'num_matches': np.sum([m['num_matches'] for m in all_metrics]),
        'num_false_positives': np.sum([m['num_false_positives'] for m in all_metrics]),
        'num_false_negatives': np.sum([m['num_false_negatives'] for m in all_metrics]),
        'id_switches': np.sum([m['id_switches'] for m in all_metrics])
    }
    
    return avg_metrics

def write_results_to_file(results, sequences, output_file):
    """Write results to a file in a well-organized format."""
    with open(output_file, 'w') as f:
        # Write header with timestamp
        f.write(f"Tracking Quality Analysis Results\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Write per-sequence results
        f.write("PER-SEQUENCE RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for seq_name in sequences:
            f.write(f"Sequence: {seq_name}\n")
            f.write("-"*40 + "\n")
            
            for method in ['ours', 'accmpeg', 'uniqp', 'aq']:
                if seq_name in results[method]:
                    metrics = results[method][seq_name]
                    f.write(f"\n{method.upper()} Method:\n")
                    f.write(f"  MOTA: {metrics['mota']:.4f}\n")
                    f.write(f"  MOTP: {metrics['motp']:.4f}\n")
                    f.write(f"  IDF1: {metrics['idf1']:.4f}\n")
                    f.write(f"  Total Ground Truth Objects: {metrics['num_gt']}\n")
                    f.write(f"  Total Predicted Objects: {metrics['num_pred']}\n")
                    f.write(f"  Total Matches: {metrics['num_matches']}\n")
                    f.write(f"  False Positives: {metrics['num_false_positives']}\n")
                    f.write(f"  False Negatives: {metrics['num_false_negatives']}\n")
                    f.write(f"  ID Switches: {metrics['id_switches']}\n")
            
            # Write comparison for this sequence
            if seq_name in results['ours'] and seq_name in results['accmpeg']:
                f.write("\nComparison:\n")
                f.write(f"  MOTA: Ours {results['ours'][seq_name]['mota']:.4f} vs Accmpeg {results['accmpeg'][seq_name]['mota']:.4f}\n")
                f.write(f"  MOTP: Ours {results['ours'][seq_name]['motp']:.4f} vs Accmpeg {results['accmpeg'][seq_name]['motp']:.4f}\n")
                f.write(f"  IDF1: Ours {results['ours'][seq_name]['idf1']:.4f} vs Accmpeg {results['accmpeg'][seq_name]['idf1']:.4f}\n")
                f.write(f"  ID Switches: Ours {results['ours'][seq_name]['id_switches']} vs Accmpeg {results['accmpeg'][seq_name]['id_switches']}\n")
            
            if seq_name in results['ours'] and seq_name in results['uniqp']:
                f.write("\nComparison:\n")
                f.write(f"  MOTA: Ours {results['ours'][seq_name]['mota']:.4f} vs Uniqp {results['uniqp'][seq_name]['mota']:.4f}\n")
                f.write(f"  MOTP: Ours {results['ours'][seq_name]['motp']:.4f} vs Uniqp {results['uniqp'][seq_name]['motp']:.4f}\n")
                f.write(f"  IDF1: Ours {results['ours'][seq_name]['idf1']:.4f} vs Uniqp {results['uniqp'][seq_name]['idf1']:.4f}\n")
                f.write(f"  ID Switches: Ours {results['ours'][seq_name]['id_switches']} vs Uniqp {results['uniqp'][seq_name]['id_switches']}\n")
            
            if seq_name in results['ours'] and seq_name in results['aq']:
                f.write("\nComparison:\n")
                f.write(f"  MOTA: Ours {results['ours'][seq_name]['mota']:.4f} vs Aq {results['aq'][seq_name]['mota']:.4f}\n")
                f.write(f"  MOTP: Ours {results['ours'][seq_name]['motp']:.4f} vs Aq {results['aq'][seq_name]['motp']:.4f}\n")
                f.write(f"  IDF1: Ours {results['ours'][seq_name]['idf1']:.4f} vs Aq {results['aq'][seq_name]['idf1']:.4f}\n")
                f.write(f"  ID Switches: Ours {results['ours'][seq_name]['id_switches']} vs Aq {results['aq'][seq_name]['id_switches']}\n")
            
            f.write("\n" + "="*80 + "\n\n")
        
        # Write overall summary
        f.write("OVERALL SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        for method in ['ours', 'accmpeg', 'uniqp', 'aq']:
            f.write(f"{method.upper()} METHOD:\n")
            f.write("-"*40 + "\n")
            
            # Calculate averages for each metric
            all_mota = [r['mota'] for r in results[method].values() if not np.isnan(r['mota'])]
            all_motp = [r['motp'] for r in results[method].values() if not np.isnan(r['motp'])]
            all_idf1 = [r['idf1'] for r in results[method].values() if not np.isnan(r['idf1'])]
            
            if all_mota:
                f.write(f"Average MOTA: {np.mean(all_mota):.4f} ± {np.std(all_mota):.4f}\n")
                f.write(f"Average MOTP: {np.mean(all_motp):.4f} ± {np.std(all_motp):.4f}\n")
                f.write(f"Average IDF1: {np.mean(all_idf1):.4f} ± {np.std(all_idf1):.4f}\n")
            
            f.write("\n")

def main():
    # Initialize YOLO model
    model = YOLO("pretrained/mot17-m.pt")
    
    # Define paths
    high_quality_base = Path("data/UNI25CHUNK")
    low_quality_base = Path("video-result")
    uniqp_base = Path("data/UNI30CHUNK")  # Directory containing UNI QP and AQ videos
    
    # Define sequences to process
    sequences = ['MOT17-02', 'MOT17-04', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
    
    # Process each sequence
    results = {
        'ours': {},
        'accmpeg': {},
        'uniqp': {},
        'aq': {}
    }
    
    for seq_name in sequences:
        print(f"\nProcessing sequence: {seq_name}")
        
        # Find the high quality video chunks in UNI25CHUNK
        high_quality_chunks = sorted(list(high_quality_base.glob(f"{seq_name}/uni*.mp4")))
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
        ours_chunks = sorted(list(seq_folder.glob("h264-ours*.mp4")))
        if not ours_chunks:
            print(f"Warning: No 'ours' chunks found for {seq_name}")
        else:
            print(f"Found {len(ours_chunks)} 'ours' chunks")
            
            # Process ours comparison
            print(f"\nComparing high quality vs 'ours' for {seq_name}")
            metrics = analyze_video_chunks(high_quality_chunks, ours_chunks, model)
            results['ours'][seq_name] = metrics
            
            print(f"Results for {seq_name} (ours):")
            print(f"MOTA: {metrics['mota']:.4f}")
            print(f"MOTP: {metrics['motp']:.4f}")
            print(f"IDF1: {metrics['idf1']:.4f}")
            print(f"ID Switches: {metrics['id_switches']}")
        
        # Get accmpeg chunks
        accmpeg_chunks = sorted(list(seq_folder.glob("h264-accmpeg*.mp4")))
        if not accmpeg_chunks:
            print(f"Warning: No 'accmpeg' chunks found for {seq_name}")
        else:
            print(f"Found {len(accmpeg_chunks)} 'accmpeg' chunks")
            
            # Process accmpeg comparison
            print(f"\nComparing high quality vs 'accmpeg' for {seq_name}")
            metrics = analyze_video_chunks(high_quality_chunks, accmpeg_chunks, model)
            results['accmpeg'][seq_name] = metrics
            
            print(f"Results for {seq_name} (accmpeg):")
            print(f"MOTA: {metrics['mota']:.4f}")
            print(f"MOTP: {metrics['motp']:.4f}")
            print(f"IDF1: {metrics['idf1']:.4f}")
            print(f"ID Switches: {metrics['id_switches']}")
        
        # Get uniform QP chunks from UNI30CHUNK
        uniqp_chunks = sorted(list(uniqp_base.glob(f"{seq_name}/uni-*.mp4")))
        if not uniqp_chunks:
            print(f"Warning: No 'uniqp' chunks found for {seq_name}")
        else:
            print(f"Found {len(uniqp_chunks)} 'uniqp' chunks")
            
            # Process uniform QP comparison
            print(f"\nComparing high quality vs 'uniqp' for {seq_name}")
            metrics = analyze_video_chunks(high_quality_chunks, uniqp_chunks, model)
            results['uniqp'][seq_name] = metrics
            
            print(f"Results for {seq_name} (uniqp):")
            print(f"MOTA: {metrics['mota']:.4f}")
            print(f"MOTP: {metrics['motp']:.4f}")
            print(f"IDF1: {metrics['idf1']:.4f}")
            print(f"ID Switches: {metrics['id_switches']}")
        
        # Get AQ chunks from UNI30CHUNK
        aq_chunks = sorted(list(uniqp_base.glob(f"{seq_name}/aq-1-*.mp4")))
        if not aq_chunks:
            print(f"Warning: No 'aq' chunks found for {seq_name}")
        else:
            print(f"Found {len(aq_chunks)} 'aq' chunks")
            
            # Process AQ comparison
            print(f"\nComparing high quality vs 'aq' for {seq_name}")
            metrics = analyze_video_chunks(high_quality_chunks, aq_chunks, model)
            results['aq'][seq_name] = metrics
            
            print(f"Results for {seq_name} (aq):")
            print(f"MOTA: {metrics['mota']:.4f}")
            print(f"MOTP: {metrics['motp']:.4f}")
            print(f"IDF1: {metrics['idf1']:.4f}")
            print(f"ID Switches: {metrics['id_switches']}")
    
    # Write results to file
    output_file = f"tracking_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    write_results_to_file(results, sequences, output_file)
    print(f"\nResults have been written to: {output_file}")

if __name__ == "__main__":
    main() 