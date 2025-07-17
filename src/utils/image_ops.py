from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms
import tempfile
import os
from PIL import Image
import subprocess
from filelock import FileLock

from src.utils.cals import closest_of_i, macroblocks_wh

QP_MAPPING = {0: 45, 1: 41, 2: 37, 3: 34, 4: 30}


def wrap_img(img: cv2.typing.MatLike, macroblock_size: int = 16) -> cv2.typing.MatLike:
    """pad the image to be divisible by macroblock size

    Args:
        img (cv2.typing.MatLike): raw image
        marcoblock_size (int): size of the macroblock

    Returns:
        cv2.typing.MatLike: padded image
    """
    h, w = img.shape[0], img.shape[1]
    h_prime = closest_of_i(h, macroblock_size)
    w_prime = closest_of_i(w, macroblock_size)
    if h != h_prime or w != w_prime:
        img = cv2.copyMakeBorder(
            img,
            0,
            h_prime - h,
            0,
            w_prime - w,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )
    return img


def retrieve_mb_img(
    img: cv2.typing.MatLike, index: int, macroblock_size: int = 16
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """retrieve the image of the specific macroblock

    Args:
        img (cv2.typing.MatLike): wrapped image
        index (int): index of the macroblock in raster scan order, start from 0
        macroblock_size (int, optional): macroblock size. Defaults to 16.

    Returns:
        np.ndarray: image of the specific macroblock
    """
    h, w = img.shape[0], img.shape[1]
    mb_w, _ = macroblocks_wh(w, h)
    mb_x = index % mb_w
    mb_y = index // mb_w
    xtl, ytl = mb_x * macroblock_size, mb_y * macroblock_size
    block = img[ytl : ytl + macroblock_size, xtl : xtl + macroblock_size]
    return block, (xtl, ytl)


def compose_img(
    img: cv2.typing.MatLike,
    index: int,
    selections: List[Tuple[int, int]],
    imageset: Dict[int, Tuple[List[np.ndarray], Tuple[int, int]]],
    macroblock_size: int = 16,
) -> cv2.typing.MatLike:
    """compose the image from the macroblocks

    Args:
        img (cv2.typing.MatLike): wrapped image
        index (List[int]): index of the macroblocks in raster scan order, start from 0
        selections (List[Tuple[int, int]]): selected qp and macroblock index
        macroblock_size (int, optional): macroblock size. Defaults to 16.

    Returns:
        np.ndarray: composed image
    """
    composed_img = img
    for mb_idx, qp in selections:
        imgs_of_qp, _ = imageset[qp]
        block, (xtl, ytl) = retrieve_mb_img(imgs_of_qp[index], mb_idx, macroblock_size)
        assert block.shape == (
            macroblock_size,
            macroblock_size,
            3,
        ), f"{block.shape} is not correct"
        composed_img[ytl : ytl + macroblock_size, xtl : xtl + macroblock_size] = block
    return composed_img


def resize_img_tensor(size: Tuple[int, int]) -> transforms.Compose:
    # normalize = transforms.Lambda(lambda x: 2 * (x / 255.0) - 1)
    tf = transforms.Compose([transforms.Resize(size)])
    return tf


def vit_transform_fn() -> transforms.Compose:
    tf = transforms.Compose(
        [
            # transforms.Resize(resize),
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return tf


def detr_transform_fn() -> transforms.Compose:
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]

    # Create the Compose transformation
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Converts PIL Image or numpy.ndarray to tensor
            transforms.Normalize(mean=image_mean, std=image_std),  # Normalizes tensor
        ]
    )
    return transform


def encode_batch_h(
    frames_path: List[str],
    indices: List[int],
    selections: List[List[Tuple[int, int]]],
    enc_tempdir: tempfile.TemporaryDirectory,
    resolutions: List[Tuple[int, int]],
    device_id: int,
    qmin: int = 30,
    qmax: int = 45,
    dataset_path: str = None,
):
    enc_tempdir_name = enc_tempdir.name

    lock = FileLock("/myh264/qp_matrix_file.lock")
    assert dataset_path is not None, "dataset path is not provided"
    with lock:
        for idx, path, selection, rs in zip(
            indices, frames_path, selections, resolutions
        ):
            tmp_sel_file = tempfile.NamedTemporaryFile()
            tem_sel_filename = tmp_sel_file.name
            with open("/myh264/qp_matrix_file", "w") as f:
                sels = [QP_MAPPING[level] for _, level in selection]
                mb_w, mb_h = macroblocks_wh(rs[1], rs[0])
                matrix = np.reshape(sels, (mb_h, mb_w))
                for row in matrix:
                    f.write(" ".join(map(str, row)) + "\n")

            dest_path = os.path.join(enc_tempdir_name, f"{idx:06d}.mp4")
            result = subprocess.run(
                [
                    "/myh264/bin/ffmpeg",
                    "-y",
                    "-i",
                    dataset_path,
                    "-start_number",
                    str(idx + 1),
                    "-vframes",
                    "1",
                    "-qp",
                    "10",
                    "-pix_fmt",
                    "yuv420p",
                    dest_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            tmp_sel_file.close()
            if result.returncode != 0:
                print(f"Error occured when encode video chunk: {result.stderr}")
                raise ValueError("Encoding failed")


def encode_batch(
    frames_path: List[str],
    indices: List[int],
    selections: List[List[Tuple[int, int]]],
    enc_tempdir: tempfile.TemporaryDirectory,
    resolutions: List[Tuple[int, int]],
    device_id: int,
    qmin: int = 30,
    qmax: int = 45,
    num_zeros: int = 6,
):
    """encode batch of raw yuv into h264 stream

    Args:
        frames_path (List[str]): yuv path
        indices (List[int]): index of this batch
        selections (List[List[Tuple[int, int]]]): qp choice of each macroblock
        enc_tempdir (tempfile.TemporaryDirectory): temporary directory for encoding
        resolutions (List[Tuple[int, int]]): resolution of each frame, should be exactly the same as the yuv

    Raises:
        ValueError: encoding failed
    """
    enc_temdir_name = enc_tempdir.name
    os.makedirs(enc_temdir_name, exist_ok=True)

    for idx, path, selection, rs in zip(indices, frames_path, selections, resolutions):
        # write selection file
        tmp_sel_file = tempfile.NamedTemporaryFile()
        tem_sel_filename = tmp_sel_file.name
        with open(tem_sel_filename, "w") as f:
            sels = []
            f.write(",".join([str(level) for _, level in selection]) + "\n")
        # shutil.copy(tem_sel_filename, "/how2compress/")
        height = rs[0]
        width = rs[1]

        if num_zeros == 6:
            dest_path = os.path.join(enc_temdir_name, f"{idx:06d}.h264")
        else:
            dest_path = os.path.join(enc_temdir_name, f"{idx:07d}.h264")

        # dest_path = os.path.join(enc_temdir_name, f"{idx:0{num_zeros}d}.h264")
        # encode
        exe = "/how2compress/src/tools/AppEncCudaEM"
        cmd = [
            exe,
            "-i",
            path,
            "-o",
            dest_path,
            "-s",
            f"{width}x{height}",
            "-gpu",
            str(device_id),
            "-e",
            tem_sel_filename,
            "-qmin",
            str(qmin),
            "-constqp",
            str(qmax),
            "-tuninginfo",
            "ultralowlatency",
            "-rc",
            "constqp",
        ]
        print(f"Running command: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error occurred when encoding video chunk:")
            print(f"Command: {' '.join(e.cmd)}")
            print(f"Return code: {e.returncode}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            raise ValueError(f"Encoding failed: {e.stderr}")
        tmp_sel_file.close()


def cv2pil(images: cv2.typing.MatLike) -> Image:
    """convert cv2 image to pil image

    Args:
        images (cv2.typing.MatLike): cv2 image

    Returns:
        Image: pil image
    """
    return Image.fromarray(cv2.cvtColor(images, cv2.COLOR_BGR2RGB))


def yuv2h264(
    yuv_frame: str,
    target_path: str,
    resolution: Tuple[int, int],
    qmin: int,
    qmax: int,
    aq_level: int,
):
    """convert yuv to h264

    Args:
        yuv_frame (str): yuv frame path
        target_path (str): target h264 path
        resolution (Tuple[int, int]): resolution of the yuv frame
        qp (int): qp level
    """
    height, width = resolution
    exe = "/how2compress/src/tools/AppEncCudaNoEM"
    subprocess.run(
        [
            exe,
            "-i",
            yuv_frame,
            "-o",
            target_path,
            "-s",
            f"{width}x{height}",
            "-qmin",
            str(qmin),
            "-qmax",
            str(qmax),
            "-aq",
            str(aq_level),
            "-initqp",
            str(qmax),
            "-gpu",
            "0",
            "-tuninginfo",
            "ultralowlatency",
            "-rc",
            "cbr",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
