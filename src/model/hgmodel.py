from transformers import MobileViTV2Model, SegformerModel
import torch
import torch.nn as nn
from einops import rearrange
from src.model.utils import ems2selections
import supervision as sv
from typing import Tuple, List
import torch.nn.functional as F
import random
import torch


class MobileVitV2(nn.Module):
    def __init__(
        self,
        pretrain_model: str = "apple/mobilevitv2-1.0-imagenet1k-256",
        threshold_inside: float = 0.7,
        threshold_outside: float = 0.75,
    ):
        super(MobileVitV2, self).__init__()
        self.model = MobileViTV2Model.from_pretrained(pretrain_model)
        self.output_size = None
        self.conv = nn.Conv2d(512, 5, 1, 1, 0)
        self.threshold_inside = threshold_inside
        self.threshold_outside = threshold_outside

    def forward(self, images):
        assert self.output_size is not None, "output size is not set"
        outputs = self.model(images)
        ems_map = F.interpolate(
            outputs.last_hidden_state, self.output_size, mode="bilinear"
        )
        # print("foward: ", ems_map[:, :2])
        ems_map = self.conv(ems_map)
        ems_map = F.softmax(ems_map, dim=1)
        _, ems_map_indices = torch.max(ems_map, dim=1)
        ems_map_indices = rearrange(ems_map_indices, "b h w -> b (h w)")
        ems_map_v = rearrange(ems_map, "b p h w -> b (h w) p")
        # print("foward: ", ems_map_v[:, :2])

        # retrieve the image
        selections = ems2selections(ems_map_indices.cpu().numpy())
        return ems_map_indices, ems_map_v, selections

    def set_output_size(self, size: Tuple[int, int]):
        self.output_size = size

    # def _auxiliary_loss(
    #     self,
    #     ems_map_indices: torch.Tensor,  # (B, N)
    #     eval_v: torch.Tensor,  # (B, N)
    #     threshold: float = 0.7,
    #     num_class: int = 5,
    # ):
    #     high_metrics_mask = eval_v > threshold
    #     adjusted_indices = torch.clamp(ems_map_indices + 1, min=0, max=num_class - 1)
    #     result = torch.where(high_metrics_mask, ems_map_indices, adjusted_indices)
    #     return result

    def loss_fn_det(
        self,
        ems_map_indices: torch.Tensor,
        ems_map_v: torch.Tensor,
        reward_mapping: torch.Tensor,
        weight_mAP: torch.Tensor,
        mAP: sv.MeanAveragePrecision,
        # ref_images: torch.Tensor,
        # compressed_images: torch.Tensor,
        # labels: List[sv.Detections],
        adjusted_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, float]:
        # loss1 guides the mAP
        loss1 = (
            -torch.log(
                torch.clamp(
                    torch.tensor(mAP.map50_95), torch.tensor(0.01), torch.tensor(0.99)
                )
            )
            * weight_mAP
        )

        # loss2 is axuliary loss for matching SSIM
        B, N, P = ems_map_v.shape
        ems_map_v = rearrange(ems_map_v, "b n p -> (b n) p")
        # assert (
        #     ref_images.shape == compressed_images.shape
        # ), f"ref images' shape {ref_images.shape} != compressed images' shape {compressed_images.shape}"
        # ssim_val, ssim_loss = mb_ssim(compressed_images, ref_images)
        # ssim_val = ssim_val.view(B, N)
        # adjusted_indices = self._auxiliary_loss(ems_map_indices, ssim_val)
        adjusted_indices = rearrange(adjusted_indices, "b n -> (b n)")
        # FIXME: using focal loss
        loss2 = (
            F.cross_entropy(
                ems_map_v, adjusted_indices, reward_mapping, reduction="mean"
            )
            * 5
        )

        return loss1 + loss2, loss1.item(), loss2.item()

    def get_adjusted_labels(
        self,
        labels: List[sv.Detections],
        emp_map_indices: torch.Tensor,
        ssim_diffs: torch.Tensor,
        mb_h: int,
        mb_w: int,
        prob: float = 0.7,
    ) -> torch.Tensor:
        """adjust the label for each marco block based on the ssim_diffs

        Args:
            labels (List[sv.Detections]): grouhdtruth labels
            emp_map_indices (torch.Tensor): qp selections of each marco block for each frame
            ssim_diffs (torch.Tensor): ssim difference of each marco block for each frame
            mb_h (int): macroblocks num in height
            mb_w (int): macroblocks num in width

        Returns:
            torch.Tensor: adjusted qp selections of each marco block for each frame. shape (B, mb_h*mb_w)
        """
        assert (
            len(labels) == emp_map_indices.shape[0] == ssim_diffs.shape[0]
        ), f"labels: {len(labels)}, emp_map_indices: {emp_map_indices.shape[0]}, ssim_diffs: {ssim_diffs.shape[0]}"
        assert (
            emp_map_indices.shape[1] == ssim_diffs.shape[1] == mb_h * mb_w
        ), f"emp_map_indices: {emp_map_indices.shape[1]}, ssim_diffs: {ssim_diffs.shape[1]}, mb_h: {mb_h}, mb_w: {mb_w}"

        b, h, w = len(labels), mb_h, mb_w
        # scale factor
        scale_factor = torch.tensor([mb_w, mb_h, mb_w, mb_h])

        # indices in/outside the bbox
        indices_in_bbox = []
        indices_outside_bbox = []

        # group mb indices with in/outside the bbox
        for label in labels:
            xyxy = label.xyxy  # (N, 4)
            xyxy = torch.from_numpy(xyxy)
            xyxy = torch.ceil(xyxy * scale_factor).int()
            # filter the raster order of index which is located in the bbox
            mask = torch.zeros((mb_h, mb_w), dtype=torch.bool)
            for bbox in xyxy:
                x1, y1, x2, y2 = bbox
                mask[y1:y2, x1:x2] = True
            mask = mask.view(-1)
            inside_indices = torch.nonzero(mask, as_tuple=False).squeeze()
            outside_indices = torch.nonzero(~mask, as_tuple=False).squeeze()
            indices_in_bbox.append(inside_indices)
            indices_outside_bbox.append(outside_indices)

        # print("indices_in_bbox: ", indices_in_bbox)
        # print("indices_outside_bbox: ", indices_outside_bbox)

        # adjust indices based on the ssim_diffs
        # for indices inside the bbox, if the ssim_diff of corresponding emp index is higher than 0.8, then keep the original index, else change to the next level by +1
        # for indices outside the bbox, if the ssim_diff of corresponding emp index is less than 0.75, then keep the original label, else change to the next level by -1
        # emp_map_indices: (B, N)
        # ssim_diffs: (B, N)
        adjusted_emp_indices = []
        for emp_indices, ssim_diff, in_indices, out_indices in zip(
            emp_map_indices, ssim_diffs, indices_in_bbox, indices_outside_bbox
        ):
            for rs_order, choice in enumerate(emp_indices):
                if (rs_order == in_indices).any():
                    if ssim_diff[rs_order] < self.threshold_inside:
                        lb = choice.item()
                        if random.random() < prob:
                            choice = torch.clamp(
                                # torch.randint(choice.item(), 5, (1,)).to(choice.device),
                                self.sample_choices(5 - choice.item(), True).to(
                                    choice.device
                                )
                                + lb,
                                min=lb,
                                max=4,
                            )
                        else:
                            choice = torch.clamp(
                                choice.flatten() + 1, min=0, max=4
                            ).flatten()
                elif (rs_order == out_indices).any():
                    if ssim_diff[rs_order] > self.threshold_outside:
                        if random.random() < prob:
                            ub = choice.item()
                            choice = torch.clamp(
                                self.sample_choices(choice.item() + 1).to(
                                    choice.device
                                ),
                                # torch.randint(0, choice.item() + 1, (1,)).to(
                                #     choice.device
                                # ),
                                min=0,
                                max=ub,
                            )
                        else:
                            choice = torch.clamp(
                                choice.flatten() - 1, min=0, max=4
                            ).flatten()
                    else:
                        choice = torch.clamp(
                            choice.flatten() + 1, min=0, max=4
                        ).flatten()
                adjusted_emp_indices.append(choice.view(-1))
        adjusted_emp_indices = (
            torch.stack(adjusted_emp_indices).view(b, h * w).contiguous()
        )
        return adjusted_emp_indices

    def noisy_adjustment(
        self,
        emp_map_indices: torch.Tensor,
        mb_h: int = 16,
        mb_w: int = 16,
        lambda_param: float = 0.4,
    ) -> torch.Tensor:
        adjusted_emp_indices = []
        for choices in emp_map_indices:
            for choice in choices:
                lb = choice.item()
                noisy_choice = lb + self.sample_choices(
                    5 - choice.item(), True, 1, lambda_param
                ).to(choice.device)
                adjusted_emp_indices.append(noisy_choice.view(-1))
        adjusted_emp_indices = (
            torch.stack(adjusted_emp_indices).view(-1, mb_h * mb_w).contiguous()
        )
        return adjusted_emp_indices

    def sample_choices(
        self,
        span: int,
        reverse: bool = False,
        num_samples: int = 1,
        lambda_param: float = 0.3,
    ) -> torch.Tensor:
        weight = torch.arange(span, dtype=torch.float32)
        if reverse:
            weight = span - weight - 1
        # use exponential distribution
        weights = torch.exp(-lambda_param * weight)
        weights = weights / weights.sum()
        sample_index = torch.multinomial(
            weights, num_samples=num_samples, replacement=True
        )
        return sample_index

    # def training_step(
    #     self,
    #     batch: Tuple[torch.Tensor, List[sv.Detections], List[int]],
    #     batch_idx: int,
    # ):
    #     images, labels, indices = batch
    #     ems_map, ems_map_v, selections = self.model(images)

    #     pass


# class Segformer(nn.Module):
#     def __init__(self, pretrained_model: str = "nvidia/mit-b0"):
#         super(Segformer, self).__init__()
#         self.model = SegformerModel.from_pretrained(pretrained_model)
#         self.up = nn.Upsample(scale_factor=2, mode="bilinear")
#         self.head = nn.Linear(256, 5)

#     def forward(self, images):
#         outputs = self.model(images)
#         ems_map = self.up(outputs.last_hidden_state)
#         ems_map = rearrange(ems_map, "b c h w -> b h w c")
#         ems_map = self.head(ems_map)
#         ems_map = F.softmax(ems_map, dim=-1)
#         ems_map_v, ems_map = torch.max(ems_map, dim=-1)
#         ems_map = rearrange(ems_map, "b h w -> b (h w)")
#         ems_map_v = rearrange(ems_map_v, "b h w -> b (h w)")
#         selections = ems2selections(ems_map.cpu().numpy())
#         return ems_map, ems_map_v, selections

#     def loss_fn_det(
#         self,
#         ems_map: torch.Tensor,
#         ems_map_v: torch.Tensor,
#         reward_mapping: torch.Tensor,
#         weight_mAP: torch.Tensor,
#         mAP: sv.MeanAveragePrecision,
#     ) -> Tuple[torch.Tensor, float, float]:
#         # loss1 = torch.sum(counts_percent * weight_counts)

#         # loss2 = torch.tensor(mAP.map50_95 - self.highest_mAP_record[seq_name])
#         loss1 = torch.exp(torch.tensor(1 - mAP.map50_95) * weight_mAP) * 0.5
#         # loss2 = F.silu(loss2) * weight_map

#         # adopt the idea from RL
#         r = reward_mapping[ems_map]
#         assert r.shape == ems_map_v.shape

#         loss2 = -torch.sum(torch.log(ems_map_v) * ems_map_v * r)

#         return loss1 + loss2, loss1.item(), loss2.item()
