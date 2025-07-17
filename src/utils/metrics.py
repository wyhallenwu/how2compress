from typing import List, Tuple

import supervision as sv


def normalize_detections(det: sv.Detections, rs: Tuple[int, int]) -> sv.Detections:
    """normalize sv.Detections for computing mAP

    Args:
        det: original sv.Detections
        rs: resolution

    Returns:
        normalized sv.Detections
    """
    width = rs[0]
    height = rs[1]
    det.xyxy[:, [0, 2]] /= width
    det.xyxy[:, [1, 3]] /= height
    return det


def compute_mAP(
    predictions: List[sv.Detections],
    frame_idxs: List[int],
    targets: List[sv.Detections],
) -> sv.MeanAveragePrecision:
    corresponding_targets = [targets[idx] for idx in frame_idxs]
    assert (
        len(predictions) == len(corresponding_targets)
    ), f"compute mAP wrong, predictions length is different from target length {len(predictions)}, {len(corresponding_targets)}"
    mAP = sv.MeanAveragePrecision.from_detections(predictions, corresponding_targets)
    return mAP


def yolo2sv(results) -> List[sv.Detections]:
    return [sv.Detections.from_ultralytics(result) for result in results]
