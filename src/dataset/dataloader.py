import os
from typing import List, Tuple, Union

import cv2
import numpy as np
import supervision as sv
import csv
import torch
from torch.utils.data import Dataset
from src.utils.load import load_h264_training
import tempfile

from src.dataset.mot_utils import get_MOT_GT, parse_seqinfo
from src.dataset.panda_utils import parse_Panda_GT, parse_panda_info, get_panda_GT
from src.dataset.visdrone_utils import get_VisDrone_GT
from src.utils import cals, image_ops
from src.utils.load import load_uniqp_imageset, lowest_quality_frame

QP_SET = [30, 34, 37, 41, 45]
QP_MAPPING = {45: 0, 41: 1, 37: 2, 34: 3, 30: 4}


def collate_fn(
    batch: List[Tuple[torch.Tensor, sv.Detections, int, List[int]]],
) -> Tuple[torch.Tensor, List[sv.Detections], List[int]]:
    """collate function for MOT dataset

    Args:
        batch (List[Tuple[cv2.typing.MatLike, sv.Detections, int]]): batch of data

    Returns:
        Tuple[torch.Tensor, List[sv.Detections], List[int]]: collated batch
    """
    images = []
    labels = []
    indices = []
    # ssim_labels = []
    for image, label, index in batch:
        images.append(image)
        labels.append(label)
        indices.append(index)
        # ssim_labels.append(torch.tensor(ssim_label))
    images = torch.stack(images).float()
    # ssim_labels = torch.stack(ssim_labels).long()
    return images, labels, indices


class MOTDataset(Dataset):
    """Dataset wrapper for MOT datasets

    The output of the __getitem__ should be:
        (wrapped raw image of given index, different quality levels of the same index image)
    """

    def __init__(
        self,
        dataset_dir: str,
        reference_dir: str,
        yuv_dir: str,
        resize_factor: int = 4,
    ):
        self.dataset_dir = dataset_dir  # MOT17Det/train
        self.reference_dir = reference_dir  # detections/
        # self.ssim_label_dir = ssim_label_dir  # ssim_labels/
        self.yuv_dir = yuv_dir  # MOT17DetYUV/
        self.labels = get_MOT_GT(dataset_dir, target_class=[1])
        self.transform_fn = image_ops.vit_transform_fn()
        self.resize_factor = resize_factor

        # dataset of current chosen sequence
        self.curr_seq = None
        self.curr_frames = None
        self.curr_labels = None
        self.curr_jpg = None
        # self.curr_uniqp_imageset = None
        # self.curr_ssim_label = None
        self.curr_yuv_frames = None
        self.curr_seq_property = {}
        # self.expected_output_size = None

    def load_sequence(self, seq_name: str):
        """update to the given sequence.
        This function should be called before fetching any data

        Args:
            seq_name (str): sequence name
        """
        self.curr_seq = seq_name
        self.curr_frames = self._load_sequence_raw(seq_name)  # path of each frame
        self.curr_labels = self._load_labels(seq_name)
        self.curr_jpg = self._load_curr_jpg(seq_name)
        # self.curr_uniqp_imageset = load_uniqp_imageset(
        #     os.path.join(self.reference_dir, self.curr_seq), QP_SET
        # )
        # self._load_ssim_label(seq_name)
        self.curr_yuv_frames = self._load_yuv_frames(seq_name)
        self.dataset_info = parse_seqinfo(os.path.join(self.dataset_dir, seq_name))
        self.curr_seq_property = {
            "name": seq_name,
            "length": len(self.curr_frames),
            "height": int(self.dataset_info["imHeight"]),
            "width": int(self.dataset_info["imWidth"]),
        }
        # w = cals.closest_of_i(self.curr_seq_property["width"], 16)
        # h = cals.closest_of_i(self.curr_seq_property["height"], 16)
        # self.transform_fn = image_ops.vit_transform_fn(
        #     (h // self.resize_factor, w // self.resize_factor)
        # )
        # self.expected_output_size = (h // 16, w // 16)

    def fetch_compose_images(
        self, indices: List[int], selections: List[List[Tuple[int, int]]]
    ) -> List[np.ndarray]:
        """fetch composed images of corresponding frames and given decisions

        Args:
            indices (List[int]): indices of batch frames
            selections (List[List[Tuple[int, int]]]): selctions of different quality levels for each frame's macroblocks

        Returns:
            List[np.ndarray]: composed images in the order of the indices
        """
        composed_images = []
        for index, selection in zip(indices, selections):
            composed_img = self._fetch_compose_img(index, selection)
            composed_images.append(composed_img)
        return composed_images

    def enc_and_ret(
        self,
        indices: List[int],
        selections: List[List[Tuple[int, int]]],
        device_id: int,
        qmin: int = 30,
        qmax: int = 45,
    ) -> Tuple[torch.Tensor, List[int]]:
        """encode the frames and return the compressed images

        Args:
            indices (List[int]): selected frames of this batch
            selections (List[List[Tuple[int, int]]]): qp decisions of each frame

        Returns:
            torch.Tensor: stacked tensor of compressed images of this batch
        """
        frames, frames_size = self._enc_and_ret(
            indices, selections, device_id, qmin, qmax
        )
        images = [image_ops.wrap_img(frame) for frame in frames]
        # images = [image_ops.cv2pil(image) for image in images]
        images = [self.transform_fn(image) for image in images]
        images = torch.stack(images).float()
        assert (
            images.shape[0] == len(indices)
        ), f"number of images {images.shape[0]} is not matching with the indices {len(indices)}"
        return images, frames_size

    def __len__(self):
        return len(self.curr_frames)

    def __getitem__(self, index: int) -> Tuple[cv2.typing.MatLike, sv.Detections, int]:
        # load the raw image
        raw_image = load_h264_training(self.curr_frames[index])
        image = image_ops.wrap_img(raw_image)
        # image = image_ops.cv2pil(image)
        image = self.transform_fn(image)
        label = self.curr_labels[index]
        # ssim_label = self.curr_ssim_label[index]
        return image, label, index

    # --------------------------------------------------------
    #                     Helper functions
    # --------------------------------------------------------
    def _load_sequence_raw(self, seq_name: str) -> List[str]:
        """helper function for loading raw frame of given sequence

        Args:
            seq_name (str): sequence name

        Returns:
            List[str]: path to each frame
        """
        # Load from resized directory instead of original sequences directory
        seq_dir = os.path.join(self.dataset_dir, f"{seq_name}_resized")
        seq_frames = sorted(os.listdir(seq_dir))
        frames_path = [os.path.join(seq_dir, frame) for frame in seq_frames]
        return frames_path

    def _load_labels(self, seq_name: str) -> List[sv.Detections]:
        """helper function for loading labels of given sequence

        Args:
            seq_name (str): sequence name

        Returns:
            List[sv.Detections]: groudtruth labels of each frame
        """
        return self.labels[seq_name]

    def _load_ssim_label(self, seq_name: str) -> List[List[int]]:
        """helper function for loading ssim labels of current sequence

        Args:
            seq_name (str): sequence name

        Returns:
            List[List[int]]: ssim labels of each frame
        """
        ssim_label_path = os.path.join(self.ssim_label_dir, seq_name + ".csv")
        ssim_label = []
        with open(ssim_label_path, "r") as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                ssim_label.append([int(x) for x in row])
        self.curr_ssim_label = ssim_label
        return ssim_label

    def _load_yuv_frames(self, seq_name: str) -> List[str]:
        """helper function for loading yuv frames of given sequence

        Args:
            seq_name (str): sequence name

        Returns:
            List[str]: path to each yuv frame
        """
        yuv_dir = os.path.join(self.yuv_dir, seq_name)
        yuv_frames = sorted(os.listdir(yuv_dir))
        yuv_frames_path = [os.path.join(yuv_dir, frame) for frame in yuv_frames]
        return yuv_frames_path

    def _load_curr_jpg(self, seq_name: str) -> List[str]:
        """helper function for loading jpg frames of given sequence

        Args:
            seq_name (str): sequence name

        Returns:
            List[str]: path to each jpg frame
        """
        jpg_dir = os.path.join(self.dataset_dir, seq_name, "img1")
        jpg_frames = sorted(os.listdir(jpg_dir))
        jpg_frames_path = [os.path.join(jpg_dir, frame) for frame in jpg_frames]
        return jpg_frames_path

    def _fetch_compose_img(
        self, index: int, selection: List[Tuple[int, int]]
    ) -> cv2.typing.MatLike:
        """helper function for fetching composed image of given frame and selection

        Args:
            index (int): index of the frame
            selection (List[int]): composition decision

        Returns:
            cv2.typing.MatLike: compsed image
        """
        mb_w, mb_h = cals.macroblocks_wh(
            self.curr_seq_property["width"], self.curr_seq_property["height"]
        )
        assert (
            len(selection) == mb_w * mb_h
        ), f"decision length {len(selection)} is not matching with num of macroblocks {mb_w * mb_h}"
        ref_img = lowest_quality_frame(self.curr_uniqp_imageset, index)
        composed_img = image_ops.compose_img(
            ref_img, index, selection, self.curr_uniqp_imageset
        )
        assert (
            composed_img.shape
            == (
                self.curr_seq_property["height"],
                self.curr_seq_property["width"],
                3,
            )
        ), f"composed image shape {composed_img.shape} is not matching with the property {self.curr_seq_property['height']}x{self.curr_seq_property['width']}"
        return composed_img

    def _enc_and_ret(
        self,
        indices: List[int],
        selections: List[List[Tuple[int, int]]],
        device_id: int,
        qmin: int = 30,
        qmax: int = 45,
    ) -> Tuple[List[cv2.typing.MatLike], List[int]]:
        frames_path = [self.curr_yuv_frames[index] for index in indices]
        # frames_path = [self.curr_jpg[index] for index in indices]
        temp_dir = tempfile.TemporaryDirectory()
        resolutions = [
            (self.curr_seq_property["height"], self.curr_seq_property["width"])
            for _ in range(len(indices))
        ]
        image_ops.encode_batch(
            frames_path,
            indices,
            selections,
            temp_dir,
            resolutions,
            device_id,
            qmin,
            qmax,
            # f"{self.dataset_dir}/{self.curr_seq}/img1/%06d.jpg",
        )
        # load frames from the temp dir
        ret_frames = []
        ret_filesize = []
        for idx in indices:
            frame_path = os.path.join(temp_dir.name, f"{idx:06d}.h264")
            frame = load_h264_training(frame_path)
            ret_frames.append(frame)
            ret_filesize.append(os.path.getsize(frame_path))
        assert (
            len(ret_frames) == len(indices)
        ), f"number of frames {len(ret_frames)} is not matching with the indices {len(indices)}"
        temp_dir.cleanup()
        return ret_frames, ret_filesize

    def enc_and_ret_val(
        self,
        indices: List[int],
        selections: List[List[Tuple[int, int]]],
        device_id: int,
        qmin: int = 30,
        qmax: int = 45,
    ) -> Tuple[torch.Tensor, List[int], List[cv2.typing.MatLike]]:
        """encode the frames and return the compressed images

        Args:
            indices (List[int]): selected frames of this batch
            selections (List[List[Tuple[int, int]]]): qp decisions of each frame

        Returns:
            torch.Tensor: stacked tensor of compressed images of this batch
        """
        frames, frames_size = self._enc_and_ret(
            indices, selections, device_id, qmin, qmax
        )
        enc_frames = frames
        images = [image_ops.wrap_img(frame) for frame in frames]
        # images = [image_ops.cv2pil(image) for image in images]
        images = [self.transform_fn(image) for image in images]
        images = torch.stack(images).float()
        assert (
            images.shape[0] == len(indices)
        ), f"number of images {images.shape[0]} is not matching with the indices {len(indices)}"
        return images, frames_size, enc_frames


class PandaDataset(Dataset):
    """Dataset wrapper for panda datasets"""

    def __init__(
        self,
        dataset_dir: str,
        gt_dir: str,
        reference_dir: str,
        yuv_dir: str,
        resize_factor: int = 4,
    ):
        self.dataset_dir = dataset_dir  # pandasRS
        self.reference_dir = reference_dir  # pandasH264/
        self.gt_dir = gt_dir  # pandas/unzipped/train_annos
        self.yuv_dir = yuv_dir  # pandasYUV/
        self.labels = get_panda_GT(self.gt_dir, target_class=[1])  # class 1 is person
        self.transform_fn = image_ops.vit_transform_fn()
        self.resize_factor = resize_factor

        # dataset of current chosen sequence
        self.curr_seq = None
        self.curr_frames = None
        self.curr_labels = None
        # self.curr_uniqp_imageset = None
        # self.curr_ssim_label = None
        self.curr_yuv_frames = None
        self.curr_seq_property = {}
        # self.expected_output_size = None

    def load_sequence(self, seq_name: str):
        """update to the given sequence.
        This function should be called before fetching any data

        Args:
            seq_name (str): sequence name
        """
        self.curr_seq = seq_name
        self.curr_frames = self._load_sequence_raw(seq_name)  # path of each frame
        self.curr_labels = self._load_labels(seq_name)
        self.curr_yuv_frames = self._load_yuv_frames(seq_name)
        self.dataset_info = parse_panda_info(os.path.join(self.gt_dir, seq_name))
        self.curr_seq_property = {
            "name": seq_name,
            "length": len(self.curr_frames),
            "height": 1440,
            "width": 2560,
        }

    def enc_and_ret(
        self,
        indices: List[int],
        selections: List[List[Tuple[int, int]]],
        device_id: int,
    ) -> Tuple[torch.Tensor, List[int]]:
        """encode the frames and return the compressed images

        Args:
            indices (List[int]): selected frames of this batch
            selections (List[List[Tuple[int, int]]]): qp decisions of each frame

        Returns:
            torch.Tensor: stacked tensor of compressed images of this batch
        """
        frames, frames_size = self._enc_and_ret(indices, selections, device_id)
        images = [image_ops.wrap_img(frame) for frame in frames]
        # images = [image_ops.cv2pil(image) for image in images]
        images = [self.transform_fn(image) for image in images]
        images = torch.stack(images).float()
        assert (
            images.shape[0] == len(indices)
        ), f"number of images {images.shape[0]} is not matching with the indices {len(indices)}"
        return images, frames_size

    def __len__(self):
        return len(self.curr_frames)

    def __getitem__(self, index: int) -> Tuple[cv2.typing.MatLike, sv.Detections, int]:
        # load the raw image
        raw_image = load_h264_training(self.curr_frames[index])
        image = image_ops.wrap_img(raw_image)
        # image = image_ops.cv2pil(image)
        image = self.transform_fn(image)
        label = self.curr_labels[index]
        # ssim_label = self.curr_ssim_label[index]
        return image, label, index

    # --------------------------------------------------------
    #                     Helper functions
    # --------------------------------------------------------
    def _load_sequence_raw(self, seq_name: str) -> List[str]:
        """helper function for loading raw frame of given sequence

        Args:
            seq_name (str): sequence name

        Returns:
            List[str]: path to each frame
        """
        seq_dir = os.path.join(self.reference_dir, seq_name)
        frame_dir = os.path.join(seq_dir, "30")
        seq_frames = sorted(os.listdir(frame_dir))
        frames_path = [os.path.join(frame_dir, frame) for frame in seq_frames]
        return frames_path

    def _load_labels(self, seq_name: str) -> List[sv.Detections]:
        """helper function for loading labels of given sequence

        Args:
            seq_name (str): sequence name

        Returns:
            List[sv.Detections]: groudtruth labels of each frame
        """
        return self.labels[seq_name]

    # def _load_ssim_label(self, seq_name: str) -> List[List[int]]:
    #     """helper function for loading ssim labels of current sequence

    #     Args:
    #         seq_name (str): sequence name

    #     Returns:
    #         List[List[int]]: ssim labels of each frame
    #     """
    #     ssim_label_path = os.path.join(self.ssim_label_dir, seq_name + ".csv")
    #     ssim_label = []
    #     with open(ssim_label_path, "r") as f:
    #         csv_reader = csv.reader(f)
    #         for row in csv_reader:
    #             ssim_label.append([int(x) for x in row])
    #     self.curr_ssim_label = ssim_label
    #     return ssim_label

    def _load_yuv_frames(self, seq_name: str) -> List[str]:
        """helper function for loading yuv frames of given sequence

        Args:
            seq_name (str): sequence name

        Returns:
            List[str]: path to each yuv frame
        """
        yuv_dir = os.path.join(self.yuv_dir, seq_name, "2560x1440")
        yuv_frames = sorted(os.listdir(yuv_dir))
        yuv_frames_path = [os.path.join(yuv_dir, frame) for frame in yuv_frames]
        return yuv_frames_path

    def _enc_and_ret(
        self,
        indices: List[int],
        selections: List[List[Tuple[int, int]]],
        device_id: int,
    ) -> Tuple[List[cv2.typing.MatLike], List[int]]:
        frames_path = [self.curr_yuv_frames[index] for index in indices]
        temp_dir = tempfile.TemporaryDirectory()
        resolutions = [
            (self.curr_seq_property["height"], self.curr_seq_property["width"])
            for _ in range(len(indices))
        ]
        image_ops.encode_batch(
            frames_path, indices, selections, temp_dir, resolutions, device_id
        )
        # load frames from the temp dir
        ret_frames = []
        ret_filesize = []
        for idx in indices:
            frame_path = os.path.join(temp_dir.name, f"{idx:06d}.h264")
            frame = load_h264_training(frame_path)
            ret_frames.append(frame)
            ret_filesize.append(os.path.getsize(frame_path))
        assert (
            len(ret_frames) == len(indices)
        ), f"number of frames {len(ret_frames)} is not matching with the indices {len(indices)}"
        temp_dir.cleanup()
        return ret_frames, ret_filesize

    def enc_and_ret_val(
        self,
        indices: List[int],
        selections: List[List[Tuple[int, int]]],
        device_id: int,
        qmin: int = 30,
        qmax: int = 45,
    ) -> Tuple[torch.Tensor, List[int], List[cv2.typing.MatLike]]:
        """encode the frames and return the compressed images

        Args:
            indices (List[int]): selected frames of this batch
            selections (List[List[Tuple[int, int]]]): qp decisions of each frame

        Returns:
            torch.Tensor: stacked tensor of compressed images of this batch
        """
        frames, frames_size = self._enc_and_ret(indices, selections, device_id)
        enc_frames = frames
        images = [image_ops.wrap_img(frame) for frame in frames]
        # images = [image_ops.cv2pil(image) for image in images]
        images = [self.transform_fn(image) for image in images]
        images = torch.stack(images).float()
        assert (
            images.shape[0] == len(indices)
        ), f"number of images {images.shape[0]} is not matching with the indices {len(indices)}"
        return images, frames_size, enc_frames


class AicityDataset(Dataset):
    """Dataset wrapper for panda datasets"""

    def __init__(
        self,
        dataset_dir: str,
        gt_dir: str,
        reference_dir: str,
        yuv_dir: str,
        resize_factor: int = 4,
    ):
        self.dataset_dir = dataset_dir  # pandasRS
        self.reference_dir = reference_dir  # pandasH264/
        self.gt_dir = gt_dir  # pandas/unzipped/train_annos
        self.yuv_dir = yuv_dir  # pandasYUV/
        self.labels = get_panda_GT(self.gt_dir, target_class=[1])  # class 1 is person
        self.transform_fn = image_ops.vit_transform_fn()
        self.resize_factor = resize_factor

        # dataset of current chosen sequence
        self.curr_seq = None
        self.curr_frames = None
        self.curr_labels = None
        # self.curr_uniqp_imageset = None
        # self.curr_ssim_label = None
        self.curr_yuv_frames = None
        self.curr_seq_property = {}
        # self.expected_output_size = None

    def load_sequence(self, seq_name: str):
        """update to the given sequence.
        This function should be called before fetching any data

        Args:
            seq_name (str): sequence name
        """
        self.curr_seq = seq_name
        self.curr_frames = self._load_sequence_raw(seq_name)  # path of each frame
        self.curr_labels = self._load_labels(seq_name)
        self.curr_yuv_frames = self._load_yuv_frames(seq_name)
        self.dataset_info = parse_panda_info(os.path.join(self.gt_dir, seq_name))
        self.curr_seq_property = {
            "name": seq_name,
            "length": len(self.curr_frames),
            "height": 1440,
            "width": 2560,
        }

    def enc_and_ret(
        self,
        indices: List[int],
        selections: List[List[Tuple[int, int]]],
        device_id: int,
    ) -> Tuple[torch.Tensor, List[int]]:
        """encode the frames and return the compressed images

        Args:
            indices (List[int]): selected frames of this batch
            selections (List[List[Tuple[int, int]]]): qp decisions of each frame

        Returns:
            torch.Tensor: stacked tensor of compressed images of this batch
        """
        frames, frames_size = self._enc_and_ret(indices, selections, device_id)
        images = [image_ops.wrap_img(frame) for frame in frames]
        # images = [image_ops.cv2pil(image) for image in images]
        images = [self.transform_fn(image) for image in images]
        images = torch.stack(images).float()
        assert (
            images.shape[0] == len(indices)
        ), f"number of images {images.shape[0]} is not matching with the indices {len(indices)}"
        return images, frames_size

    def __len__(self):
        return len(self.curr_frames)

    def __getitem__(self, index: int) -> Tuple[cv2.typing.MatLike, sv.Detections, int]:
        # load the raw image
        raw_image = load_h264_training(self.curr_frames[index])
        image = image_ops.wrap_img(raw_image)
        # image = image_ops.cv2pil(image)
        image = self.transform_fn(image)
        label = self.curr_labels[index]
        # ssim_label = self.curr_ssim_label[index]
        return image, label, index

    # --------------------------------------------------------
    #                     Helper functions
    # --------------------------------------------------------
    def _load_sequence_raw(self, seq_name: str) -> List[str]:
        """helper function for loading raw frame of given sequence

        Args:
            seq_name (str): sequence name

        Returns:
            List[str]: path to each frame
        """
        seq_dir = os.path.join(self.reference_dir, seq_name)
        frame_dir = os.path.join(seq_dir, "30")
        seq_frames = sorted(os.listdir(frame_dir))
        frames_path = [os.path.join(frame_dir, frame) for frame in seq_frames]
        return frames_path

    def _load_labels(self, seq_name: str) -> List[sv.Detections]:
        """helper function for loading labels of given sequence

        Args:
            seq_name (str): sequence name

        Returns:
            List[sv.Detections]: groudtruth labels of each frame
        """
        return self.labels[seq_name]

    # def _load_ssim_label(self, seq_name: str) -> List[List[int]]:
    #     """helper function for loading ssim labels of current sequence

    #     Args:
    #         seq_name (str): sequence name

    #     Returns:
    #         List[List[int]]: ssim labels of each frame
    #     """
    #     ssim_label_path = os.path.join(self.ssim_label_dir, seq_name + ".csv")
    #     ssim_label = []
    #     with open(ssim_label_path, "r") as f:
    #         csv_reader = csv.reader(f)
    #         for row in csv_reader:
    #             ssim_label.append([int(x) for x in row])
    #     self.curr_ssim_label = ssim_label
    #     return ssim_label

    def _load_yuv_frames(self, seq_name: str) -> List[str]:
        """helper function for loading yuv frames of given sequence

        Args:
            seq_name (str): sequence name

        Returns:
            List[str]: path to each yuv frame
        """
        yuv_dir = os.path.join(self.yuv_dir, seq_name, "2560x1440")
        yuv_frames = sorted(os.listdir(yuv_dir))
        yuv_frames_path = [os.path.join(yuv_dir, frame) for frame in yuv_frames]
        return yuv_frames_path

    def _enc_and_ret(
        self,
        indices: List[int],
        selections: List[List[Tuple[int, int]]],
        device_id: int,
    ) -> Tuple[List[cv2.typing.MatLike], List[int]]:
        frames_path = [self.curr_yuv_frames[index] for index in indices]
        temp_dir = tempfile.TemporaryDirectory()
        resolutions = [
            (self.curr_seq_property["height"], self.curr_seq_property["width"])
            for _ in range(len(indices))
        ]
        image_ops.encode_batch(
            frames_path, indices, selections, temp_dir, resolutions, device_id
        )
        # load frames from the temp dir
        ret_frames = []
        ret_filesize = []
        for idx in indices:
            frame_path = os.path.join(temp_dir.name, f"{idx:06d}.h264")
            frame = load_h264_training(frame_path)
            ret_frames.append(frame)
            ret_filesize.append(os.path.getsize(frame_path))
        assert (
            len(ret_frames) == len(indices)
        ), f"number of frames {len(ret_frames)} is not matching with the indices {len(indices)}"
        temp_dir.cleanup()
        return ret_frames, ret_filesize

    def enc_and_ret_val(
        self,
        indices: List[int],
        selections: List[List[Tuple[int, int]]],
        device_id: int,
        qmin: int = 30,
        qmax: int = 45,
    ) -> Tuple[torch.Tensor, List[int], List[cv2.typing.MatLike]]:
        """encode the frames and return the compressed images

        Args:
            indices (List[int]): selected frames of this batch
            selections (List[List[Tuple[int, int]]]): qp decisions of each frame

        Returns:
            torch.Tensor: stacked tensor of compressed images of this batch
        """
        frames, frames_size = self._enc_and_ret(indices, selections, device_id)
        enc_frames = frames
        images = [image_ops.wrap_img(frame) for frame in frames]
        # images = [image_ops.cv2pil(image) for image in images]
        images = [self.transform_fn(image) for image in images]
        images = torch.stack(images).float()
        assert (
            images.shape[0] == len(indices)
        ), f"number of images {images.shape[0]} is not matching with the indices {len(indices)}"
        return images, frames_size, enc_frames


class VisDroneDataset(Dataset):
    """Dataset wrapper for VisDrone datasets"""

    def __init__(
        self,
        dataset_dir: str,
        reference_dir: str,
        yuv_dir: str,
        resize_factor: int = 4,
    ):
        self.dataset_dir = dataset_dir  # VisDrone sequences directory
        self.reference_dir = reference_dir  # VisDroneH264/
        self.yuv_dir = yuv_dir  # VisDroneYUV/
        self.labels = get_VisDrone_GT(dataset_dir, target_class=[0])  # class 0 is person
        self.transform_fn = image_ops.vit_transform_fn()
        self.resize_factor = resize_factor

        # dataset of current chosen sequence
        self.curr_seq = None
        self.curr_frames = None
        self.curr_labels = None
        self.curr_yuv_frames = None
        self.curr_seq_property = {}

    def load_sequence(self, seq_name: str):
        """update to the given sequence.
        This function should be called before fetching any data

        Args:
            seq_name (str): sequence name
        """
        self.curr_seq = seq_name
        self.curr_frames = self._load_sequence_raw(seq_name)  # path of each frame
        self.curr_labels = self._load_labels(seq_name)
        self.curr_yuv_frames = self._load_yuv_frames(seq_name)
        
        # Get image dimensions from first frame
        first_frame = cv2.imread(self.curr_frames[0])
        height, width = first_frame.shape[:2]
        
        self.curr_seq_property = {
            "name": seq_name,
            "length": len(self.curr_frames),
            "height": height,
            "width": width,
        }

    def enc_and_ret(
        self,
        indices: List[int],
        selections: List[List[Tuple[int, int]]],
        device_id: int,
        qmin: int = 30,
        qmax: int = 45,
    ) -> Tuple[torch.Tensor, List[int]]:
        """encode the frames and return the compressed images

        Args:
            indices (List[int]): selected frames of this batch
            selections (List[List[Tuple[int, int]]]): qp decisions of each frame

        Returns:
            torch.Tensor: stacked tensor of compressed images of this batch
        """
        frames, frames_size = self._enc_and_ret(
            indices, selections, device_id, qmin, qmax
        )
        images = [image_ops.wrap_img(frame) for frame in frames]
        images = [self.transform_fn(image) for image in images]
        images = torch.stack(images).float()
        assert (
            images.shape[0] == len(indices)
        ), f"number of images {images.shape[0]} is not matching with the indices {len(indices)}"
        return images, frames_size

    def __len__(self):
        return len(self.curr_frames)

    def __getitem__(self, index: int) -> Tuple[cv2.typing.MatLike, sv.Detections, int]:
        # load the raw image
        raw_image = load_h264_training(self.curr_frames[index])
        image = image_ops.wrap_img(raw_image)
        image = self.transform_fn(image)
        label = self.curr_labels[index]
        return image, label, index

    # --------------------------------------------------------
    #                     Helper functions
    # --------------------------------------------------------
    def _load_sequence_raw(self, seq_name: str) -> List[str]:
        """helper function for loading raw frame of given sequence

        Args:
            seq_name (str): sequence name

        Returns:
            List[str]: path to each frame
        """
        # Load from resized directory with correct path pattern
        seq_dir = os.path.join(self.dataset_dir, "sequences", f"{seq_name}_resized")
        seq_frames = sorted(os.listdir(seq_dir))
        frames_path = [os.path.join(seq_dir, frame) for frame in seq_frames]
        return frames_path

    def _load_labels(self, seq_name: str) -> List[sv.Detections]:
        """helper function for loading labels of given sequence

        Args:
            seq_name (str): sequence name

        Returns:
            List[sv.Detections]: groudtruth labels of each frame
        """
        return self.labels[seq_name]

    def _load_yuv_frames(self, seq_name: str) -> List[str]:
        """helper function for loading yuv frames of given sequence

        Args:
            seq_name (str): sequence name

        Returns:
            List[str]: path to each yuv frame
        """
        yuv_dir = os.path.join(self.yuv_dir, seq_name)
        yuv_frames = sorted(os.listdir(yuv_dir))
        yuv_frames_path = [os.path.join(yuv_dir, frame) for frame in yuv_frames]
        return yuv_frames_path

    def _enc_and_ret(
        self,
        indices: List[int],
        selections: List[List[Tuple[int, int]]],
        device_id: int,
        qmin: int = 30,
        qmax: int = 45,
    ) -> Tuple[List[cv2.typing.MatLike], List[int]]:
        frames_path = [self.curr_yuv_frames[index] for index in indices]
        temp_dir = tempfile.TemporaryDirectory()
        resolutions = [
            (self.curr_seq_property["height"], self.curr_seq_property["width"])
            for _ in range(len(indices))
        ]
        # print(f"frame path: {frames_path}")
        # print(f"temp_dir: {temp_dir.name}, resolutions: {resolutions}")
        # print(f"selections: {selections}")
        print(f"indices: {indices}")
        image_ops.encode_batch(
            frames_path,
            indices,
            selections,
            temp_dir,
            resolutions,
            device_id,
            qmin,
            qmax,
            7
        )
        # load frames from the temp dir
        ret_frames = []
        ret_filesize = []
        for idx in indices:
            frame_path = os.path.join(temp_dir.name, f"{idx:07d}.h264")
            frame = load_h264_training(frame_path)
            ret_frames.append(frame)
            ret_filesize.append(os.path.getsize(frame_path))
        assert (
            len(ret_frames) == len(indices)
        ), f"number of frames {len(ret_frames)} is not matching with the indices {len(indices)}"
        temp_dir.cleanup()
        return ret_frames, ret_filesize

    def enc_and_ret_val(
        self,
        indices: List[int],
        selections: List[List[Tuple[int, int]]],
        device_id: int,
        qmin: int = 30,
        qmax: int = 45,
    ) -> Tuple[torch.Tensor, List[int], List[cv2.typing.MatLike]]:
        """encode the frames and return the compressed images

        Args:
            indices (List[int]): selected frames of this batch
            selections (List[List[Tuple[int, int]]]): qp decisions of each frame

        Returns:
            torch.Tensor: stacked tensor of compressed images of this batch
        """
        # print(f"selections: {selections}")
        frames, frames_size = self._enc_and_ret(indices, selections, device_id, qmin, qmax)
        enc_frames = frames
        images = [image_ops.wrap_img(frame) for frame in frames]
        # images = [image_ops.cv2pil(image) for image in images]
        images = [self.transform_fn(image) for image in images]
        images = torch.stack(images).float()
        assert (
            images.shape[0] == len(indices)
        ), f"number of images {images.shape[0]} is not matching with the indices {len(indices)}"
        return images, frames_size, enc_frames
