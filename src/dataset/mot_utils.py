import configparser
import os
from typing import Dict, List, Tuple

import numpy as np
import supervision as sv
from tqdm import tqdm

from src.utils.metrics import normalize_detections


def parse_seqinfo(dir: str) -> configparser.SectionProxy:
    """parse the seqinfo.ini file

    Args:
        dir (str): directory of the MOT dataset to the seqinfo.ini file

    Returns:
        configparser.SectionProxy: the section of the seqinfo.ini
    """
    config = configparser.ConfigParser()
    seqinfo_file = os.path.join(dir, "seqinfo.ini")
    config.read(seqinfo_file)
    assert "Sequence" in config, f"parse {seqinfo_file} error"
    return config["Sequence"]


def parse_MOT(
    dir: str,
    origin_size: Tuple[int, int],
    tgt_obj_id: List[int],
    frame_num: int,
    filter: bool = False,
) -> List[sv.Detections]:
    """parse MOT groundtruth

    Args:
        dir (str): directory of the MOT dataset
        origin_size (Tuple[int, int]): resolution of the original image
        tgt_obj_id (List[int]): objects of interest
        frame_num (int): number of frames
        filter (bool, optional): filter the target object or not. Defaults to False.

    Returns:
        List[sv.Detections]: groundtruth of each frame
    """
    gt_file = os.path.join(dir, "gt/gt.txt")
    ow, oh = origin_size
    boxes = [[sv.Detections.empty()] for _ in range(frame_num)]
    with open(gt_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            fields = line.strip().split(",")
            idx, class_idx = int(fields[0]), int(fields[7])
            x, y, w, h = (
                float(fields[2]),
                float(fields[3]),
                float(fields[4]),
                float(fields[5]),
            )
            if class_idx not in tgt_obj_id:
                continue

            # filter the box partially out of image
            if filter and (x < 0 or y < 0 or (x + w) > ow or (y + h) > oh):
                continue

            # normalize the gt bboxes
            xtl = x
            ytl = y
            xrb = x + w
            yrb = y + h

            detections = sv.Detections(
                xyxy=np.array([[xtl, ytl, xrb, yrb]]),
                class_id=np.array([class_idx]),
                data={},
            )
            detections = normalize_detections(detections, (ow, oh))
            boxes[idx - 1].append(detections)
    boxes = [sv.Detections.merge(dets) for dets in boxes]
    assert (
        len(boxes) == frame_num
    ), f"parsing MOT groudtruth error, {len(boxes)} is different from {frame_num}"
    return boxes


def get_MOT_GT(src_dir: str, target_class: List[int]) -> Dict[str, List[sv.Detections]]:
    """get MOT groundtruth

    Args:
        src_dir (str): source directory of the MOT dataset
        target_class (List[int]): object of interest

    Returns:
        Dict[str, List[sv.Detections]]: groundtruth of each sequence, key is the sequence name, value is the groundtruth of each frame
    """

    # MOT20Det/train/(MOT20-02)/(gt, img1, seqinfo.ini)

    sub_folders = os.listdir(src_dir)
    results = {}
    for sub_folder in tqdm(sub_folders, desc="get_MOT_gt"):
        dir = os.path.join(src_dir, sub_folder)
        seqinfo = parse_seqinfo(dir)
        rs = (int(seqinfo["imWidth"]), int(seqinfo["imHeight"]))
        frame_num = int(seqinfo["seqLength"])
        gt_detections = parse_MOT(
            dir,
            rs,
            target_class,
            frame_num,
        )
        results[sub_folder] = gt_detections
    return results
