import torch
import torch.nn as nn
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    DetrForObjectDetection,
    DetrImageProcessor,
    YolosForObjectDetection,
)
from PIL import Image
import torch.nn.functional as F
from einops import rearrange
from src.utils import image_ops, load
import os
import numpy as np
from tqdm import tqdm
import requests

DATASET = [
    # "MOT17-02",
    "MOT17-04",
    # "MOT17-09",
    # "MOT17-10",
    # "MOT17-11",
    # "MOT17-13",
    # "MOT20-02",
]


# # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# # image = Image.open(requests.get(url, stream=True).raw)
# image = Image.open("/how2compress/data/MOT17Det/train/MOT17-04/img1/000001.jpg")


class DetrInferencer(nn.Module):
    def __init__(self):
        super(DetrInferencer, self).__init__()
        self.preprocessor = image_ops.vit_transform_fn()
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50", revision="no_timm"
        )

    def forward(self, x, resize_factor=4):
        h, w = x.shape[2], x.shape[3]
        x = F.interpolate(x, (h // resize_factor, w // resize_factor))
        return self.model(x)


# processor = DetrImageProcessor.from_pretrained(
#     "facebook/detr-resnet-50", revision="no_timm"
# )
# test_input = torch.rand(1, 3, 1088, 1920)
# test_input = image_ops.vit_transform_fn()(image).unsqueeze(0)
# print(test_input.shape)
# # test_input = processor(test_input, return_tensors="pt", do_rescale=False)
# model = DetrForObjectDetection.from_pretrained(
#     "facebook/detr-resnet-50", revision="no_timm"
# )
# model.eval()
# # inputs = processor(images=image, return_tensors="pt")
# target_sizes = torch.tensor([image.size[::-1]])
# # print(target_sizes.shape)
# results = model(test_input)
# print(results.logits.shape)
# print(results.pred_boxes.shape)

# results = processor.post_process_object_detection(results, target_sizes=target_sizes)[0]
# # print(results)
# for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#     box = [round(i, 2) for i in box.tolist()]
#     print(
#         f"Detected {model.config.id2label[label.item()]} with confidence "
#         f"{round(score.item(), 3)} at location {box}"
#     )


def get_accgrad(
    high_quality_path: str, low_quality_path: str, tgt_path: str, threshold: float = 5
):
    # high_quality_path = "/how2compress/data/MOT17DetH264/{dataset}/30/"
    # low_quality_path = "/how2compress/data/MOT17DetH264/{dataset}/45/"
    model = DetrInferencer().to("cuda").eval()
    hq_images = sorted(os.listdir(high_quality_path))
    hq_images = [os.path.join(high_quality_path, img) for img in hq_images]
    lq_images = sorted(os.listdir(low_quality_path))
    lq_images = [os.path.join(low_quality_path, img) for img in lq_images]

    tgt_accgrad = []

    for hq_img, lq_img in tqdm(zip(hq_images, lq_images)):
        hq_img = load.load_h264_training(hq_img)
        hq_img = image_ops.wrap_img(hq_img)
        hq_img = image_ops.cv2pil(hq_img)
        hq_img = model.preprocessor(hq_img).unsqueeze(0)
        h, w = hq_img.shape[2], hq_img.shape[3]

        lq_img = load.load_h264_training(lq_img)
        lq_img = image_ops.wrap_img(lq_img)
        lq_img = image_ops.cv2pil(lq_img)
        lq_img = model.preprocessor(lq_img).unsqueeze(0)
        lq_img = lq_img.requires_grad_()

        output_hq = model(hq_img.to("cuda"))
        output_lq = model(lq_img.to("cuda"))
        loss1 = F.mse_loss(output_lq.logits, output_hq.logits, reduction="mean")
        loss2 = F.mse_loss(output_lq.pred_boxes, output_hq.pred_boxes, reduction="mean")
        loss = loss1 + loss2
        loss.backward()
        accgrad = F.interpolate(lq_img.grad, size=(h // 16, w // 16))
        # print(accgrad)
        # print(accgrad.shape)
        accgrad = torch.max(accgrad, dim=1).values.flatten()
        mini = accgrad.min()
        maxi = accgrad.max()
        th = torch.quantile(accgrad, threshold).item()
        # accgrad = (accgrad - mini) / (maxi - mini)
        mask = accgrad > th
        mask = mask.to(torch.int)
        # print(mask.shape)
        tgt_accgrad.append(mask.cpu().numpy().tolist())
    tgt = np.array(tgt_accgrad)
    zero_count = (np.size(tgt) - np.count_nonzero(tgt)) / np.size(tgt)
    print(zero_count)
    np.savetxt(tgt_path, tgt, fmt="%d")


if __name__ == "__main__":
    for dataset in DATASET:
        get_accgrad(
            f"/how2compress/data/MOT17DetH264/{dataset}/30/",
            f"/how2compress/data/MOT17DetH264/{dataset}/45/",
            f"/how2compress/data/accmpeg_gt/detr-{dataset}-{0.7}q.txt",
            0.7,
        )
