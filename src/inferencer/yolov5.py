from typing import Any, List, Tuple

import cv2
import supervision as sv
import torch
from paper.inferencer.base import BaseInferencer

# all supported models of yolov5
ModelList = ["n", "s", "m", "l", "x"]


class YOLOV5Inferencer(BaseInferencer):
    """yolov5 inferencer based on ultralytics yolov5"""

    def __init__(self, model_size: str, device: str = "cpu"):
        super(BaseInferencer, self).__init__()
        self.model_size = model_size
        self.device = device
        self.model = self._setup_model()

    def _setup_model(self) -> Any:
        """private method to setup the model

        Returns:
            ultralytics yolov5 model
        """
        model_name = "yolov5" + self.model_size

        assert (
            self.model_size in ModelList
        ), f"{self.model_size} is not supported by ultralytics yolov5"

        model = torch.hub.load("ultralytics/yolov5", model_name)
        # TODO: load the model from local dir
        # model = torch.hub.load(
        #     "./yolov5", "custom", f"{model_name}.pt", source="local", verbose=False
        # )
        return model.to(self.device)

    def predict(
        self, frames: List[Any], imgsz: Tuple[int, int] = (1080, 1920)
    ) -> List[sv.Detections]:
        """predict a batch of frames

        Args:
            frames: a list of frame type which is supported by ultralytics yolov5

        Returns:
            a list of supervision Detections
        """
        results = self.model(frames)
        results = [sv.Detections.from_yolov5(result) for result in results.tolist()]
        return results


if __name__ == "__main__":
    # test unit

    frame = cv2.imread("./dataset/zidane.jpg")
    frames = [frame for _ in range(10)]
    v5_inferencer = YOLOV5Inferencer("x")

    results = v5_inferencer.predict(frames)
    print(results)
