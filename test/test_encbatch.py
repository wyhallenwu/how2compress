from src.utils.image_ops import encode_batch
import random
import tempfile
import shutil
import os
import torch

frames_path = [
    "/how2compress/data/MOT17DetYUV/MOT17-04/000001.yuv",
    "/how2compress/data/MOT17DetYUV/MOT17-04/000002.yuv",
]

indices = [1, 2]
selections = [
    [(i, random.randint(0, 3)) for i in range(8160)],
    [(i, random.randint(0, 3)) for i in range(8160)],
]
s1 = torch.tensor([v for _, v in selections[0]])
s2 = torch.tensor([v for _, v in selections[1]])
print(torch.bincount(s1, minlength=5))
print(torch.bincount(s2, minlength=5))
resolutions = [(1080, 1920), (1080, 1920)]

tempdir = tempfile.TemporaryDirectory()
encode_batch(frames_path, indices, selections, tempdir, resolutions)
name = tempdir.name
for file in os.listdir(name):
    shutil.copy(os.path.join(name, file), "./")
