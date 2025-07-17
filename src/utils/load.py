import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Tuple

import cv2
import numpy as np

from src.utils.image_ops import wrap_img


def load_frames(
    path: str,
) -> Tuple[List[cv2.typing.MatLike], int, Tuple[int, int]]:
    """load dataset with only raw frames, the resolution of all frames must be the same

    Args:
        path: the path to the frames

    Returns:
        ret: list of frames
        len: number of frames
        resolution: (height, width) of the frames

    """
    files = os.listdir(path)
    files = [os.path.join(path, file) for file in files]

    ret = []
    for file in files:
        frame = cv2.imread(file)
        height, width = frame.shape[0], frame.shape[1]
        ret.append(frame)

    return ret, len(ret), (height, width)


def load_video(
    path: str,
) -> Tuple[List[cv2.typing.MatLike], int, Tuple[int, int]]:
    """load video and return the captured frames

    Args:
        path: path to the video file

    Returns:
        ret: list of frames
        len: number of frames
        resolution: (height, width) of the frames
    """
    cap = cv2.VideoCapture(path)
    ret = []
    assert cap.isOpened(), "video capture error"
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            ret.append(frame)
        else:
            break
    cap.release()
    return ret, len(ret), (height, width)


def load_h264_training(path: str) -> cv2.typing.MatLike:
    """load the h264 file for training, each file contains only one frame

    Args:
        path (str): path to the h264 file

    Returns:
        cv2.typing.MatLike: frame
    """
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), "video capture error"
    success, frame = cap.read()
    assert success, "video read error"
    cap.release()
    return frame


def load_mp4_training(path: str) -> cv2.typing.MatLike:
    """load the h264 file for training, each file contains only one frame

    Args:
        path (str): path to the h264 file

    Returns:
        cv2.typing.MatLike: frame
    """
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), "video capture error"
    success, frame = cap.read()
    assert success, "video read error"
    cap.release()
    return frame


def load_uniqp_imageset(
    src_dir: str, qp_set: List[int], max_workers: int = 16
) -> Dict[int, Tuple[List[np.ndarray], Tuple[int, int]]]:
    """load images of certain QP from the dataset

    Args:
        src_dir (str): source directory of the dataset
        qp_set (List[int]): list of chosen QP
        max_workers (int): max number of workers used for IO

    Returns:
        Dict[str, Tuple[List[np.ndarray], Tuple[int, int]]]: imageset of each QP
    """

    # this multi-threading function is not really faster than the single-threading one
    # we just keep this for potential improvement
    def load_frames_fn(
        path: str,
        qp: int,
    ) -> Tuple[List[cv2.typing.MatLike], int, Tuple[int, int], int]:
        start_t = time.time()
        files = sorted(os.listdir(path))
        files = [os.path.join(path, file) for file in files]

        ret = [None] * len(files)
        with ThreadPoolExecutor(max_workers=32) as pe:
            # index is for sorting the completed futures
            futures = {
                pe.submit(load_h264_training, file): i for i, file in enumerate(files)
            }
            for future in as_completed(futures):
                index = futures[future]
                frame = future.result()
                frame = wrap_img(frame)
                height, width = frame.shape[:2]
                ret[index] = frame
        assert all(frame is not None for frame in ret), "Some frames are missing!"
        end_t = time.time()
        # print(f"load {qp} time: {end_t - start_t}")

        return ret, len(ret), (height, width), qp

    # def load_frames_fn(
    #     path: str,
    #     qp: int,
    # ) -> Tuple[List[cv2.typing.MatLike], int, Tuple[int, int], int]:
    #     start_t = time.time()
    #     files = os.listdir(path)
    #     files = [os.path.join(path, file) for file in files]

    #     ret = []
    #     cap = cv2.VideoCapture()
    #     for file in files:
    #         # frame = cv2.imread(file)
    #         cap.open(file)
    #         _, frame = cap.read()
    #         height, width = frame.shape[0], frame.shape[1]
    #         frame = wrap_img(frame)
    #         ret.append(frame)

    #     end_t = time.time()
    #     print(f"load {qp} time: {end_t - start_t}")
    #     return ret, len(ret), (height, width), qp

    image_set = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pe:
        futures = [
            pe.submit(load_frames_fn, os.path.join(src_dir, str(qp)), qp)
            for qp in qp_set
        ]
        for future in as_completed(futures):
            frames, _, (height, width), qp = future.result()
            image_set[qp] = (frames, (height, width))
    return image_set


def lowest_quality_frame(
    imageset: Dict[int, Tuple[List[np.ndarray], Tuple[int, int]]],
    idx: int,
    lowest_qp: int = 45,
) -> np.ndarray:
    frames, _ = imageset[lowest_qp]
    return frames[idx]
