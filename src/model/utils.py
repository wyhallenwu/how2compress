import warnings
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

QP_MAP = {
    0: 45,
    1: 41,
    2: 37,
    3: 34,
    4: 30,
}


def resize(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    warning=True,
):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Upsample(nn.Module):
    def __init__(
        self, size=None, scale_factor=None, mode="nearest", align_corners=None
    ):
        super(Upsample, self).__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if not self.size:
            size = [int(t * self.scale_factor) for t in x.shape[-2:]]
        else:
            size = self.size
        return resize(x, size, None, self.mode, self.align_corners)


def ems2selections(ems: np.ndarray) -> List[List[Tuple[int, int]]]:
    """convert ems to selections

    Args:
        ems (np.ndarray): ems of the sequence

    Returns:
        List[List[Tuple[int, int]]]: selections of different quality levels for each frame's macroblocks
    """
    selections = []
    for em in ems:
        selection = []
        for i, qp in enumerate(em):
            selection.append((i, qp))
        selections.append(selection)
    return selections
