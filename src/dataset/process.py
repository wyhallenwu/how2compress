import os
import shutil
import subprocess
import tempfile
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from src.utils.cals import make_even
from src.utils.load import load_video

RESOLUTION = [(1920, 1080), (1600, 900), (1280, 720), (960, 540), (640, 480)]
QP = [30, 34, 37, 41, 45]
DATASET = ["1702YUV", "1704YUV", "1713YUV"]
MOT = {"1702YUV": "MOT1702", "1704YUV": "MOT1704", "1713YUV": "MOT1713"}


# -----------------------------------------------------------------
#                     convert frame to YUV format
# -----------------------------------------------------------------


def frames_tmp_dir(
    src_dir: str, frames_idxs: List[int], num_zeros: int = 6
) -> tempfile.TemporaryDirectory[str]:
    """copy corresponding frames to the temp directory and returns the context manager of that temp dir

    Args:
        src_dir: source directory of the frames
        frames_idxs: selected frames indice

    Returns:
        context manager the temp dir
    """
    tmp = tempfile.TemporaryDirectory()
    tmp_dir = tmp.name
    for idx, frame_idx in enumerate(frames_idxs):
        # source = os.path.join(src_dir, f"{frame_idx:06d}.jpg")
        # dest = os.path.join(tmp_dir, f"{idx:06d}.jpg")
        source = os.path.join(src_dir, f"{frame_idx:0{num_zeros}d}.jpg")
        dest = os.path.join(tmp_dir, f"{idx:0{num_zeros}d}.jpg")
        shutil.copy(source, dest)
    return tmp


def convert2yuv(src_dir: str, des_dir: str, frames_per_chunk: int = 30, num_zeros: int = 6):
    """convert frames to YUV format in chunks

    Args:
        src_dir (str): source directory of frames
        des_dir (str): save directory of YUV files
        frames_per_chunk (int, optional): num of frames in each chunk. Defaults to 30.
    """

    # src_dir: [000001.jpg...]
    num_frames = len(os.listdir(src_dir))
    num_chunks = num_frames // frames_per_chunk

    for i in tqdm(range(num_chunks)):
        with frames_tmp_dir(
            src_dir,
            list(range(i * frames_per_chunk + 1, (i + 1) * frames_per_chunk + 1)),
            num_zeros,
        ) as td:
            dest = os.path.join(des_dir, f"{(i+1):0{num_zeros}d}.yuv")
            if not os.path.exists(des_dir):
                os.makedirs(des_dir)
            yuv_frames = []
            for file in sorted(os.listdir(td)):
                img_path = os.path.join(td, file)
                frame = cv2.imread(img_path)
                h, w = frame.shape[:2]
                # make sure the h and w is even number, or it will raise error
                if h % 2 != 0 or w % 2 != 0:
                    frame = cv2.resize(frame, (make_even(w), make_even(h)))
                yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
                yuv_frames.append(yuv_frame)
            yuv_frames = np.array(yuv_frames)
            with open(dest, "wb") as f:
                f.write(yuv_frames.tobytes())


# -----------------------------------------------------------------
#                      encode YUV to raw h264
# -----------------------------------------------------------------


def encode_chunk(
    filename: str,
    dest_path: str,
    width: int,
    height: int,
    base_qp: int,
    qmin: int,
    mb_qp_file: str | None,
):
    """encode video chunk of YUV format to raw h264 bitstream

    Args:
        filename (str): source file path
        dest_path (str): destination raw h264 bitstream
        width (int): width of the resolution
        height (int): height of the resolution
        base_qp (int): const qp
        qmin (int): minimum qp
        mb_qp_file (str): macroblock QP delta map file path if None, encode with uniform QP
    """
    if mb_qp_file is None:
        exe = "/how2compress/src/tools/AppEncCudaNoEM"
        result = subprocess.run(
            [
                exe,
                "-i",
                filename,
                "-o",
                dest_path,
                "-s",
                f"{width}x{height}",
                # "-fps",
                # fps,
                "-qmin",
                str(qmin),
                "-qmax",
                str(base_qp),
                "-constqp",
                str(base_qp),
                "-tuninginfo",
                "ultralowlatency",
                "-rc",
                "constqp",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    else:
        exe = "/how2compress/src/tools/AppEncCudaEM"
        result = subprocess.run(
            [
                exe,
                "-i",
                filename,
                "-o",
                dest_path,
                "-s",
                f"{width}x{height}",
                "-e",
                mb_qp_file,
                # "-fps",
                # fps,
                "-qmin",
                str(qmin),
                "-qmax",
                str(base_qp),
                "-constqp",
                str(base_qp),
                "-tuninginfo",
                "ultralowlatency",
                "-rc",
                "constqp",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    if result.returncode != 0:
        print(f"Error occured when encode video chunk: {result.stderr}")


def encode_chunks(
    source_folder: str,
    dest_folder: str,
    width: int,
    height: int,
    base_qp: int,
    qmin: int,
    mb_qp_file: str | None,
):
    chunks = sorted(os.listdir(source_folder))
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    for chunk in tqdm(chunks, desc=f"{source_folder}|{base_qp}"):
        filename = os.path.join(source_folder, chunk)
        dest_filename = os.path.join(dest_folder, chunk.split(".")[0] + ".h264")
        encode_chunk(filename, dest_filename, width, height, base_qp, qmin, mb_qp_file)


# -----------------------------------------------------------------
#                     decode the raw h264 to jpg
# -----------------------------------------------------------------


def decode_chunks(src_dir: str, dest_dir: str) -> int:
    """decode the raw h264 to jpg

    Args:
        src_dir (str): source directory of the raw h264
        dest_dir (str): destination directory of the jpg

    Returns:
        int: num of frames decoded in total
    """
    chunks = sorted(os.listdir(src_dir))
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    index = 0
    for chunk in tqdm(chunks, desc=f"{src_dir}"):
        filename = os.path.join(src_dir, chunk)
        frames, _, _ = load_video(filename)
        for frame in frames:
            # index start from 1
            index += 1
            dest_filename = os.path.join(dest_dir, f"{index:06d}.jpg")
            cv2.imwrite(dest_filename, frame)

    return index
