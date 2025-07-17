from ultralytics import YOLO
from pathlib import Path

def main():
    # Initialize YOLO model
    model = YOLO("yolo11m.pt")
    
    # Define paths
    high_quality_base = Path("/how2compress/data/VisDrone-UNI25-MP4")  # QP25 videos as high quality
    low_quality_base = Path("/how2compress/data/VisDrone-UNI30-MP4")   # QP30 videos for uniform QP and spatial AQ
    results_base = Path("/how2compress/results/visdrone")              # Results for our method and AccMPEG
    
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
        
        # Get uniform QP chunks (QP30)
        uniqp_chunks = sorted(list(low_quality_base.glob(f"{seq_name}_resized/uni*.mp4")))
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
        
        # Get spatial AQ chunks (QP30 with AQ mode 1)
        spatial_aq_chunks = sorted(list(low_quality_base.glob(f"{seq_name}_resized/aq-1-*.mp4")))
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