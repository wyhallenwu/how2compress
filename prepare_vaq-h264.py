import os
from src.utils import image_ops
from src.dataset.mot_utils import parse_seqinfo
from tqdm import tqdm
import subprocess
import math

DATASET = ["MOT17-02", "MOT17-04", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]
# DATASET = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]

mot_root = "/how2compress/data/MOT17Det/train"
dest = "/how2compress/data/MOT17-VAQ-H264"
frames = 25
qp = 30
aq_mode = 0

# ffmpeg -start_number 100 -i input_%03d.jpg -vframes 200 -framerate 25 -c:v libx264 -qp 30 -x264-params aq-mode=0 output.mp4

for dataset in DATASET:
    if not os.path.exists(os.path.join(dest, dataset)):
        os.makedirs(os.path.join(dest, dataset))

    src_path = os.path.join(mot_root, dataset, "img1")
    files = os.listdir(src_path)
    num = len(files)
    num_chunks = math.ceil(num / frames)

    for i in tqdm(range(num_chunks)):
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                os.path.join(src_path, "%06d.jpg"),
                "-start_number",
                str(i * frames + 1),
                "-vframes",
                str(frames),
                "-framerate",
                "25",
                "-c:v",
                "libx264",
                "-qp",
                str(qp),
                "-x264-params",
                f"aq-mode={aq_mode}",
                os.path.join(dest, dataset, f"{i+1:06d}.mp4"),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
