from typing import List, Tuple

import supervision as sv
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.dataset.dataloader import MOTDataset
from src.model.segformer import MixVisionTransformer
from src.model.utils import ems2selections


class EncodePipeline(nn.Module):
    def __init__(self):
        super(EncodePipeline, self).__init__()
        self.model = MixVisionTransformer(
            patch_size=16, in_chans=3, num_classes=5, depths=[1, 2, 3, 2]
        )

    # def register_inference_model(self, inferencer: nn.Module):
    #     self.inferencer = inferencer
    #     for param in self.inferencer.parameters():
    #         param.requires_grad = False

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[Tuple[int, int]]]]:
        ems_map = self.model(x)
        # ems_map = torch.argmax(ems_map, dim=-1)
        ems_map_v, ems_map = torch.max(ems_map, dim=-1)
        ems_map = rearrange(ems_map, "b h w -> b (h w)")
        ems_map_v = rearrange(ems_map_v, "b h w -> b (h w)")

        # retrieve the image
        selections = ems2selections(ems_map.cpu().numpy())
        return ems_map, ems_map_v, selections

    def loss_fn_det(
        self,
        ems_map: torch.Tensor,
        ems_map_v: torch.Tensor,
        reward_mapping: torch.Tensor,
        counts_percent: torch.Tensor,
        weight_counts: torch.Tensor,
        weight_mAP: torch.Tensor,
        mAP: sv.MeanAveragePrecision,
    ) -> Tuple[torch.Tensor, float, float, float]:
        loss1 = torch.sum(counts_percent * weight_counts)

        # loss2 = torch.tensor(mAP.map50_95 - self.highest_mAP_record[seq_name])
        loss2 = torch.tensor(1 - mAP.map50_95) * weight_mAP
        # loss2 = F.silu(loss2) * weight_map

        # adopt the idea from RL
        r = reward_mapping[ems_map]
        assert r.shape == ems_map_v.shape
        loss3 = -torch.mean(torch.log(ems_map_v) * r)

        return loss1 + loss2 + loss3, loss1.item(), loss2.item(), loss3.item()
