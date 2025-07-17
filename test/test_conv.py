import math
import time
from typing import List

import torch
import torch.nn as nn


class DownsampleNet(nn.Module):
    def __init__(
        self, input_channels: int = 3, output_channels: List[int] = [16, 32, 64, 128]
    ):
        super(DownsampleNet, self).__init__()

        self.layer1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels[0],
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.layer2 = nn.Conv2d(
            in_channels=output_channels[0],
            out_channels=output_channels[1],
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.layer3 = nn.Conv2d(
            in_channels=output_channels[1],
            out_channels=output_channels[2],
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.layer4 = nn.Conv2d(
            in_channels=output_channels[2],
            out_channels=output_channels[3],
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))  # Downsample to (16, H/2, W/2)
        x = self.relu(self.layer2(x))  # Downsample to (32, H/4, W/4)
        x = self.relu(self.layer3(x))  # Downsample to (64, H/8, W/8)
        x = self.relu(self.layer4(x))  # Downsample to (C, H/16, W/16)
        return x


class LargeKernelDownsampleNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=128):
        super(LargeKernelDownsampleNet, self).__init__()

        self.layer1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=64,
            kernel_size=7,
            stride=4,
            padding=3,
        )
        self.layer2 = nn.Conv2d(
            in_channels=64,
            out_channels=output_channels,
            kernel_size=7,
            stride=4,
            padding=3,
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))  # Downsample to (64, H/4, W/4)
        x = self.relu(self.layer2(x))  # Downsample to (C, H/16, W/16)
        return x


model = LargeKernelDownsampleNet().to("cuda:0")
H = 1080
W = 1920
warmup_dummy = [torch.randn(1, 3, H, W).to("cuda:0") for _ in range(10)]
for dummy in warmup_dummy:
    model(dummy)

x = torch.randn(1, 3, H, W).to("cuda:0")
start_t = time.time()
y = model(x)
end_t = time.time()
print(f"Time: {end_t - start_t}")
print(y.shape)
assert y.shape == (1, 128, math.ceil(H / 16), math.ceil(W / 16))
