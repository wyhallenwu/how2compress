import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification
from PIL import Image
import torch.nn.functional as F
from einops import rearrange
from src.utils import image_ops, load
import os
import numpy as np
from tqdm import tqdm
from einops import rearrange
from src.model.utils import ems2selections


class AccMpeg(nn.Module):
    def __init__(self, h, w, choices=2):
        super(AccMpeg, self).__init__()
        self.model = AutoModelForImageClassification.from_pretrained(
            "google/mobilenet_v2_1.0_224"
        )
        self.model.classifier = nn.Linear(1280, choices * h * w, bias=True)
        self.label_root = "/how2compress/data/accmpeg_gt"
        self.h = h
        self.w = w
        self.choices = choices
        self.label = None

    def forward(self, x):
        output = self.model(x)
        output = rearrange(
            output.logits, "b (c h w) -> b c h w", h=self.h, w=self.w, c=self.choices
        )
        ems_map = F.softmax(output, dim=1)
        _, ems_map_indices = torch.max(ems_map, dim=1)
        ems_map_indices = rearrange(ems_map_indices, "b h w -> b (h w)")
        ems_map_v = rearrange(ems_map, "b p h w -> b (h w) p")
        selections = ems2selections(ems_map_indices.cpu().numpy())
        offset_selections = []
        for selection in selections:
            offset_selection = []
            for idx, choice in selection:
                if choice == 1:
                    offset_selection.append((idx, 4))
                else:
                    offset_selection.append((idx, choice))
            offset_selections.append(offset_selection)
        return ems_map_indices, ems_map_v, offset_selections
    
    def forward_fg(self, x):
        output = self.model(x)
        output = rearrange(
            output.logits, "b (c h w) -> b c h w", h=self.h, w=self.w, c=self.choices
        )
        ems_map = F.softmax(output, dim=1)

    def load_labels(self, dataset: str):
        path = os.path.join(self.label_root, f"{dataset}-0.6q.txt")
        label = np.loadtxt(path, delimiter=" ", dtype=int)
        label = torch.from_numpy(label).to(torch.long)
        self.label = label

    def loss_fn_det(
        self, ems_map_v: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        ems_map_v = rearrange(ems_map_v, "b n p -> (b n) p")
        labels = labels.flatten()
        # print(ems_map_v.shape, labels.shape)
        loss = F.cross_entropy(ems_map_v, labels, reduction="mean")
        return loss
