from typing import Any, List, Tuple

import supervision as sv
import torch
import torch.nn as nn
from src.inferencer.base import BaseInferencer
from ultralytics import YOLO
import cv2


class YOLOV8Inferencer(BaseInferencer, nn.Module):
    def __init__(self, model_path: str):
        super(BaseInferencer, self).__init__()
        nn.Module.__init__(self)
        self.model_path = model_path
        self.model = self._setup_model()
        self._freeze_model()

    def _setup_model(self) -> Any:
        """private method to setup the model

        Returns:
            the ultralytics yolov8 model
        """
        model = YOLO(self.model_path, verbose=False)
        return model

    def _freeze_model(self):
        for param in self.parameters():
            param.requires_grad = False

    def predict(
        self, frames: List[Any], imgsz: Tuple[int, int] = (1080, 1920)
    ) -> List[sv.Detections]:
        """predict a batch of frames.

        Args:
            frames: list of any frame type that is supported by ultralytics yolov8

        Returns:
            list of supervision Detections
        """
        results = self.model.predict(frames, imgsz=imgsz, verbose=False, classes=[1])
        results = [sv.Detections.from_ultralytics(result) for result in results]
        return results


if __name__ == "__main__":
    v8_inferencer = YOLOV8Inferencer(
        "/how2compress/pretrained/best_MOT_1920.pt"
    )
    frame = cv2.imread("/how2compress/data/MOT17Det/train/MOT17-04/img1/000001.jpg")
    frames = [frame]
    results = v8_inferencer.predict(frames)
    print(results)
