from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from supervision import Detections


class BaseInferencer(ABC):
    """The base interface for all inferencer"""

    @abstractmethod
    def predict(self, frames: List[Any], imgsz=Tuple[int, int]) -> List[Detections]:
        """predict batch of frames"""
        pass

    @abstractmethod
    def _setup_model(self) -> Any:
        """setup and initialize the model"""
        pass
