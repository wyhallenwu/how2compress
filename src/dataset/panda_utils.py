import json
import os
import supervision as sv
import numpy as np
from typing import List, Dict
from tqdm import tqdm


def parse_panda_info(gt_path: str):
    info_path = os.path.join(gt_path, "seqinfo.json")
    with open(info_path, "r") as f:
        info = json.load(f)
    # print(info["seqLength"])
    return info


def parse_Panda_GT(gt_path: str) -> List[sv.Detections]:
    seqinfo = parse_panda_info(gt_path)
    tracks_info = os.path.join(gt_path, "tracks.json")
    # print(seqinfo["seqLength"])

    detections = [[sv.Detections.empty()] for _ in range(seqinfo["seqLength"])]
    with open(tracks_info, "r") as f:
        gts = json.load(f)
        # print(gts[0])
        for gt in gts:
            for frame in gt["frames"]:
                xtl = frame["rect"]["tl"]["x"]
                ytl = frame["rect"]["tl"]["y"]
                xbr = frame["rect"]["br"]["x"]
                ybr = frame["rect"]["br"]["y"]
                if (
                    xtl < 0
                    or ytl < 0
                    or xbr < 0
                    or ybr < 0
                    or xtl > 1
                    or ytl > 1
                    or xbr > 1
                    or ybr > 1
                ):
                    continue
                detections[frame["frame id"] - 1].append(
                    sv.Detections(
                        class_id=np.array([1]),
                        xyxy=np.array([[xtl, ytl, xbr, ybr]]),
                    )
                )
    detections = [sv.Detections.merge(dets) for dets in detections]
    return detections


def get_panda_GT(
    src_dir: str,
    target_class: List[int] = [1],
) -> Dict[str, List[sv.Detections]]:
    # /how2compress/data/pandas/unzipped/train_annos
    sub_folders = os.listdir(src_dir)
    results = {}
    for sub_folder in tqdm(sub_folders, desc="get_Panda_gt"):
        dir = os.path.join(src_dir, sub_folder)
        detections = parse_Panda_GT(dir)
        results[sub_folder] = detections
    return results


if __name__ == "__main__":
    parse_Panda_GT(
        "/how2compress/data/pandas/unzipped/train_annos/01_University_Canteen/"
    )
