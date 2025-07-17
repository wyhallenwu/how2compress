from typing import Any, Dict, List, Tuple

import numpy as np

# [qp, (raw_frame, (height, width))]
ImageSet = Dict[int, Tuple[List[np.ndarray], Tuple[int, int]]]
