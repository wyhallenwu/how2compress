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

DATASET = [
    "MOT17-02",
    "MOT17-04",
    "MOT17-09",
    "MOT17-10",
    "MOT17-11",
    "MOT17-13",
    # "MOT20-02",
]


class AccMpeg(nn.Module):
    def __init__(self, num_classes=2):
        super(AccMpeg, self).__init__()
        self.model = AutoModelForImageClassification.from_pretrained(
            "google/mobilenet_v2_1.0_224"
        )
        self.model.classifier = nn.Linear(1280, num_classes, bias=True)

    def forward(self, x):
        output = self.model(x)
        output = F.softmax(output.logits, dim=-1)
        return output


class Inferencer(nn.Module):
    def __init__(self):
        super(Inferencer, self).__init__()
        self.preprocessor = image_ops.detr_transform_fn()
        # self.model = DetrForObjectDetection.from_pretrained(
        #     "facebook/detr-resnet-50", revision="no_timm"
        # )
        self.model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")

    def forward(self, x, resize_factor=4):
        h, w = x.shape[2], x.shape[3]
        x = F.interpolate(x, (h // resize_factor, w // resize_factor))
        return self.model(x)


def get_accgrad(
    high_quality_path: str, low_quality_path: str, tgt_path: str, threshold: float = 0.5
):
    # high_quality_path = "/how2compress/data/MOT17DetH264/{dataset}/30/"
    # low_quality_path = "/how2compress/data/MOT17DetH264/{dataset}/45/"
    model = Inferencer().to("cuda").eval()
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
        loss1 = F.mse_loss(output_lq.logits, output_hq.logits, reduction="sum")
        # loss2 = F.mse_loss(output_lq.pred_boxes, output_hq.pred_boxes, reduction="mean")
        loss = loss1
        loss.backward()
        accgrad = F.interpolate(lq_img.grad, size=(h // 16, w // 16))
        # print(accgrad)
        # print(accgrad.shape)
        accgrad = torch.max(accgrad, dim=1).values.flatten()
        # mini = accgrad.min()
        # maxi = accgrad.max()
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
    # model = Inferencer().to("cuda")
    # x = torch.randn(1, 3, 1088, 1920).to("cuda")
    # y = torch.randn(1, 3, 1088, 1920).to("cuda")

    # x = F.interpolate(x, size=(1088 // 4, 1920 // 4), mode="bilinear")
    # y = F.interpolate(y, size=(1088 // 4, 1920 // 4), mode="bilinear")
    # x = x.requires_grad_()
    # print(x.shape)
    # # print(model)
    # x_logits = model(x).logits
    # x_pred_boxes = model(x).pred_boxes
    # y_logits = model(y).logits
    # y_pred_boxes = model(y).pred_boxes
    # loss1 = F.mse_loss(x_logits, y_logits, reduction="mean")
    # loss2 = F.mse_loss(x_pred_boxes, y_pred_boxes, reduction="mean")
    # loss = loss1 + loss2
    # print(loss.item())
    # loss.backward()
    # accgrad = x.grad
    # accgrad = F.interpolate(accgrad, size=(1088 // 16, 1920 // 16))
    # accgrad = torch.max(accgrad, dim=1).values
    # print(accgrad)
    # print(accgrad.shape)
    for dataset in DATASET:
        get_accgrad(
            f"/how2compress/data/MOT17DetH264/{dataset}/30/",
            f"/how2compress/data/MOT17DetH264/{dataset}/45/",
            f"/how2compress/data/accmpeg_gt/{dataset}-{0.6}q.txt",
            0.6,
        )
    # model = Inferencer()
    # frame1 = load.load_h264_training(
    #     "/how2compress/data/MOT17DetH264/MOT17-04/30/000001.h264"
    # )
    # frame2 = load.load_h264_training(
    #     "/how2compress/data/MOT17DetH264/MOT17-02/45/000002.h264"
    # )
    # frame1 = image_ops.wrap_img(frame1)
    # frame2 = image_ops.wrap_img(frame2)
    # x = model.preprocessor(frame1).unsqueeze(0)
    # print(x.size)
    # x = x.requires_grad_()
    # y = model.preprocessor(frame2).unsqueeze(0)

    # print(f"x shape: {x.shape}, y shape: {y.shape}")
    # output_a = model(x)
    # print(output_a.logits.shape)
    # print(output_a.pred_boxes.shape)
    # output_b = model(y)
    # print(output_b.logits.shape)
    # print(output_b.pred_boxes.shape)
    # loss = F.mse_loss(output_a.logits, output_b.logits, reduction="sum")
    # loss.backward()
    # print(x.grad.shape)
    # # print(x.grad)
    # accgrad = F.interpolate(x.grad, size=(1088 // 16, 1920 // 16))
    # accgrad = torch.max(accgrad, dim=1).values.flatten()
    # mask = accgrad > 0.2
    # mask = mask.to(torch.int)
    # print(mask)
    # print(mask.shape)
    # print(len(mask.numpy().tolist()))
    # # y = torch.sum(output)
    # # y.backward()
    # # print(dummy_a.requires_grad, dummy_b.requires_grad)
    # # print(output)
    # # print(dummy_a.grad.shape)
