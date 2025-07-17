import math
from typing import Tuple
import torch
from pytorch_msssim import ssim, SSIM
from einops import rearrange


def closest_of_i(length: int, i: int = 32) -> int:
    """find the closest larger value which could be divided by i

    Args:
        length (int): given value
        i (int, optional): denominator. Defaults to 32.

    Returns:
        int: closest larger value
    """
    if length % i == 0:
        return length
    else:
        return (length // i + 1) * i


def make_even(value: int) -> int:
    """make the value even

    Args:
        value (int): initial value

    Returns:
        int: even value which is closest to the initial value
    """
    return value if value % 2 == 0 else value - 1


def macroblocks_wh(
    width: int, height: int, macroblock_size: int = 16
) -> Tuple[int, int]:
    """generate the macroblocks' width and height

    Args:
        width (int): width of the image
        height (int): height of the image
        macroblock_size (int): size of the macroblock

    Returns:
        List[Tuple[int, int]]: list of the macroblocks' width and height
    """

    mb_width = math.ceil(width / macroblock_size)
    mb_height = math.ceil(height / macroblock_size)
    return (mb_width, mb_height)


def mb_ssim(
    composed_images: torch.Tensor,
    reference_images: torch.Tensor,
    mb_blocksize: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """calculate the SSIM of each macroblock

    Args:
        composed_images (torch.Tensor): the composed image
        reference_images (torch.Tensor): the reference image
        mb_blocksize (int, optional): size of the macroblock. Defaults to 16.

    Returns:
        torch.Tensor: SSIM of each macroblock
        torch.Tensor: SSIM loss
    """
    B = composed_images.shape[0]
    composed_images = rearrange(
        composed_images,
        "b c (h mb1) (w mb2) -> (b h w) c mb1 mb2",
        mb1=mb_blocksize,
        mb2=mb_blocksize,
    )
    reference_images = rearrange(
        reference_images,
        "b c (h mb1) (w mb2) -> (b h w) c mb1 mb2",
        mb1=mb_blocksize,
        mb2=mb_blocksize,
    )
    assert (
        torch.max(composed_images) <= 1.0 and torch.max(reference_images) <= 1.0
    ), f"composed_images: {torch.max(composed_images)}, reference_images: {torch.max(reference_images)}"
    ssim_val = ssim(
        reference_images, composed_images, data_range=1.0, size_average=False
    ).contiguous()
    ssim_val = ssim_val.view(B, -1)
    ssim_module = SSIM(data_range=1.0, size_average=True, channel=3)
    ssim_loss = 1 - ssim_module(reference_images, composed_images)
    return ssim_val, ssim_loss


def get_indices(tensor: torch.Tensor, ref_diff: float = 0.3):
    # Get the top two values and their indices for each column
    top2_values, top2_indices = torch.topk(tensor, k=2, dim=0)

    # Calculate the difference between the top two values
    diff = top2_values[0] - top2_values[1]

    # Create a mask where the difference is less than 0.1
    mask = diff < ref_diff

    # Select either the top index or the second-top index based on the mask
    result_indices = torch.where(mask, top2_indices[1], top2_indices[0])

    return result_indices
