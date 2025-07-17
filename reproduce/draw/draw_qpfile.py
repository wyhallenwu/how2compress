import numpy as np
import cv2

qp = [30, 34, 38, 42, 46]
qp = [30, 40]

with open("/myh264/qp_matrix_file", "w") as f:
    # matrix = np.ones((68, 120), dtype=int) * 35
    # matrix = np.random.randint(30, 46, (68, 120), dtype=int)
    matrix = np.random.choice(qp, (68, 120), replace=True)
    for row in matrix:
        f.write(" ".join(map(str, row)) + "\n")


frame = cv2.imread("/how2compress/data/MOT17Det/train/MOT17-04/img1/000001.jpg")
frame = cv2.resize(frame, (1920, 1088))
cv2.imwrite("/myh264/raw/000001.jpg", frame)
