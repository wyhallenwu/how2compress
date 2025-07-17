from typing import Any, List, Tuple

import supervision as sv
import torch
from src.inferencer.base import BaseInferencer
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor
import cv2

MODEL_LIST = [
    "50",
    "101",
]


class DETRInferencer(BaseInferencer):
    def __init__(self, model_size: str):
        super(BaseInferencer, self).__init__()
        self.model_size = model_size
        self.model, self.processor = self._setup_model()

    def _setup_model(self):
        """helper function to setup the model

        Returns:
            detr model and corresponding processor
        """

        assert (
            self.model_size in MODEL_LIST
        ), f"detr-resnet-{self.model_size} is not supported, choose one from {MODEL_LIST}"
        model = DetrForObjectDetection.from_pretrained(
            f"facebook/detr-resnet-{self.model_size}", revision="no_timm"
        )
        processor = DetrImageProcessor.from_pretrained(
            f"facebook/detr-resnet-{self.model_size}", revision="no_timm"
        )
        return model, processor

    def predict(self, frames: List[Any], imgsz: Tuple[int, int]) -> List[sv.Detections]:
        """predict a batch of frames

        Args:
            frames: list of PIL.Image
            resolution: tuple of int: height, width

        Returns:
            list of supervision Detections
        """
        # inputs = self.processor(images=frames, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(frames)
        target_sizes = torch.tensor([[imgsz[0], imgsz[1]] for _ in range(len(frames))]).to("cuda:1")
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes
        )

        rets = []
        for result in results:
            indices = torch.nonzero(result["labels"] == 1, as_tuple=True)[0]
            labels = result["labels"][indices]
            scores = result["scores"][indices]
            boxes = result["boxes"][indices]
            ret = {"scores": scores, "labels": labels, "boxes": boxes}
            rets.append(ret)

        return [
            sv.Detections.from_transformers(
                transformers_results=result, id2label=self.model.config.id2label
            )
            for result in rets 
        ]


if __name__ == "__main__":
    detr_inferencer = DETRInferencer("101")
    # frame = Image.open("/how2compress/data/MOT17Det/train/MOT17-04/img1/000001.jpg")
    frame = cv2.imread("/how2compress/data/MOT17Det/train/MOT17-04/img1/000001.jpg")
    frames = [frame for _ in range(2)]
    results = detr_inferencer.predict(frames, imgsz=(1080, 1920))
    print(type(results))
    print(results)
