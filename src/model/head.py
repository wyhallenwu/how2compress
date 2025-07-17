from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.model.utils import resize


class UniMLP(nn.Module):
    def __init__(self, in_chan: int, out_chan: int):
        super(UniMLP, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.linear_proj = nn.Linear(self.in_chan, self.out_chan)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.linear_proj(x)
        return x


class SegHead(nn.Module):
    def __init__(self, in_chans: List[int], num_classes: int, hidden_dim: int):
        super(SegHead, self).__init__()
        self.in_chans = in_chans
        c1_chan, c2_chan, c3_chan = self.in_chans

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.linear_pred = nn.Conv2d(
            self.hidden_dim * len(in_chans), self.hidden_dim, 1
        )
        self.norm = nn.BatchNorm2d(self.hidden_dim)
        self.seg_head = nn.Linear(self.hidden_dim, self.num_classes)

        self.linear_c1 = UniMLP(c1_chan, self.hidden_dim)
        self.linear_c2 = UniMLP(c2_chan, self.hidden_dim)
        self.linear_c3 = UniMLP(c3_chan, self.hidden_dim)
        # self.linear_c4 = UniMLP(c4_chan, self.hidden_dim)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        assert len(x) == 3, "input feature maps should be 3 levels"
        c1, c2, c3 = x
        # _, _, h, w = c4.shape
        # _c4 = self.linear_c4(c4)
        # _c4 = rearrange(_c4, "b (h w) c -> b c h w", h=h, w=w)
        # _c4 = resize(_c4, size=c1.size()[2:], mode="bilinear", align_corners=False)

        _, _, h, w = c3.shape
        _c3 = self.linear_c3(c3)
        _c3 = rearrange(_c3, "b (h w) c -> b c h w", h=h, w=w)
        _c3 = resize(_c3, size=c1.size()[2:], mode="bilinear", align_corners=False)

        _, _, h, w = c2.shape
        _c2 = self.linear_c2(c2)
        _c2 = rearrange(_c2, "b (h w) c -> b c h w", h=h, w=w)
        _c2 = resize(_c2, size=c1.size()[2:], mode="bilinear", align_corners=False)

        _, _, h, w = c1.shape
        _c1 = self.linear_c1(c1)
        _c1 = rearrange(_c1, "b (h w) c -> b c h w", h=h, w=w)
        _c1 = resize(_c1, size=c1.size()[2:], mode="bilinear", align_corners=False)

        out = torch.cat([_c1, _c2, _c3], dim=1)
        # out = rearrange(out, 'b c h w -> b h w c')
        out = self.linear_pred(out)
        out = self.norm(out)
        out = rearrange(out, "b c h w -> b h w c")
        out = self.seg_head(out)
        out = F.softmax(out, dim=-1)
        return out
