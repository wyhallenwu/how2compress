import torch
import torch.nn as nn
from src.model.hgmodel import MobileVitV2
from src.utils import image_ops, load

model = MobileVitV2()
model.load_state_dict(
    torch.load("/how2compress/pretrained/train/1722937152.2452426/0-best.pth")
)
model.to("cuda:2")
frames = [
    "/how2compress/data/MOT17DetH264/MOT17-02/30/000001.h264",
    "/how2compress/data/MOT17DetH264/MOT17-02/30/000002.h264",
]
print(model)
transform = image_ops.vit_transform_fn()
images = [transform(load.load_h264_training(path)) for path in frames]
images = torch.stack(images).to("cuda:2")
print(images.shape)
ems_map_indices, ems_map_v, selections = model(images)
bincounts = torch.bincount(ems_map_indices.flatten(), minlength=5)
print(bincounts)
