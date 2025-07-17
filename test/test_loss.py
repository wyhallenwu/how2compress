import math
import numpy as np

x = [round(i, 2) for i in np.arange(0.6, 0.7, 0.01)]
for p in x:
    print(-math.log(p) * 10)
