import os
import numpy as np
import supervision as sv
from typing import Dict, List, Tuple
from tqdm import tqdm
from src.utils.metrics import normalize_detections
import cv2

def parse_VisDrone_GT(
    gt_file: str,
    origin_size: Tuple[int, int],
    tgt_obj_id: List[int],
    frame_num: int,
    filter: bool = False,
) -> List[sv.Detections]:
    """Parse VisDrone groundtruth

    Args:
        gt_file (str): Path to the groundtruth file
        origin_size (Tuple[int, int]): Resolution of the original image (1344x756)
        tgt_obj_id (List[int]): Objects of interest
        frame_num (int): Number of frames
        filter (bool, optional): Filter the target object or not. Defaults to False.

    Returns:
        List[sv.Detections]: Groundtruth of each frame
    """
    ow, oh = origin_size
    # Calculate scaling factors for new resolution (1920x1080)
    scale_x = 1920 / ow
    scale_y = 1080 / oh
    
    # Map VisDrone class ID to model class ID
    class_id_map = {1: 0}  # Map VisDrone person (1) to model person (0)
    
    boxes = [[sv.Detections.empty()] for _ in range(frame_num)]
    
    with open(gt_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            fields = line.strip().split(",")
            frame_id = int(fields[0])
            visdrone_class_id = int(fields[7])  # VisDrone class ID is in field 7
            
            # Map the class ID if it exists in the mapping
            class_id = class_id_map.get(visdrone_class_id, visdrone_class_id)
            
            if class_id not in tgt_obj_id:
                continue

            x, y, w, h = (
                float(fields[2]),
                float(fields[3]),
                float(fields[4]),
                float(fields[5]),
            )
            
            # Scale coordinates to new resolution
            x = x * scale_x
            y = y * scale_y
            w = w * scale_x
            h = h * scale_y

            # Filter the box partially out of image
            if filter and (x < 0 or y < 0 or (x + w) > 1920 or (y + h) > 1080):
                continue

            # Convert to xyxy format for supervision
            xtl = x
            ytl = y
            xrb = x + w
            yrb = y + h

            detections = sv.Detections(
                xyxy=np.array([[xtl, ytl, xrb, yrb]]),
                class_id=np.array([class_id]),
                data={},
            )
            detections = normalize_detections(detections, (1920, 1080))
            boxes[frame_id - 1].append(detections)
            
    boxes = [sv.Detections.merge(dets) for dets in boxes]
    assert len(boxes) == frame_num, f"Parsing VisDrone groundtruth error, {len(boxes)} is different from {frame_num}"
    return boxes

def get_VisDrone_GT(src_dir: str, target_class: List[int]) -> Dict[str, List[sv.Detections]]:
    """Get VisDrone groundtruth

    Args:
        src_dir (str): Source directory of the VisDrone dataset (e.g. /how2compress/data/visdrone/VisDrone2019-VID-val)
        target_class (List[int]): Objects of interest

    Returns:
        Dict[str, List[sv.Detections]]: Groundtruth of each sequence, key is the sequence name, value is the groundtruth of each frame
    """
    # Get all annotation files
    annotations_dir = os.path.join(src_dir, "annotations")
    sequences_dir = os.path.join(src_dir, "sequences")
    
    results = {}
    
    # Get all annotation files
    annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.txt')]
    
    for annotation_file in tqdm(annotation_files, desc="get_VisDrone_gt"):
        # Get sequence name from annotation filename
        seq_name = os.path.splitext(annotation_file)[0]
        seq_dir = os.path.join(sequences_dir, seq_name)
        gt_file = os.path.join(annotations_dir, annotation_file)
        
        # Use original resolution (1344x756)
        width, height = 1344, 756
        
        # Count number of frames
        frame_num = len([f for f in os.listdir(seq_dir) if f.endswith('.jpg')])
        
        gt_detections = parse_VisDrone_GT(
            gt_file,
            (width, height),
            target_class,
            frame_num,
        )
        results[seq_name] = gt_detections
        
    return results 