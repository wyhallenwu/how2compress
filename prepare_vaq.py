import os
from src.utils import image_ops
from src.dataset.mot_utils import parse_seqinfo
from tqdm import tqdm

DATASET = ["MOT17-02", "MOT17-04", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]
# DATASET = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]

mot_root = "/how2compress/data/MOT17Det/train"
root = "/how2compress/data/MOT17DetYUV"
qp = "30"

target = "/how2compress/data/MOT17H264VAQ0-2"

if not os.path.exists(target):
    os.makedirs(target)

for dataset in DATASET:
    path = os.path.join(root, dataset)
    yuv_frames = sorted(os.listdir(path))
    yuv_frames = [os.path.join(path, yuv_frame) for yuv_frame in yuv_frames]

    seq_info = parse_seqinfo(os.path.join(mot_root, dataset))
    width, height = int(seq_info["imWidth"]), int(seq_info["imHeight"])

    tgt_root = os.path.join(target, dataset)
    if not os.path.exists(tgt_root):
        os.makedirs(tgt_root)
    for i, yuv_frame in tqdm(enumerate(yuv_frames), desc=f"Converting {dataset}"):
        save_file = os.path.join(tgt_root, f"{i+1:06d}.h264")
        image_ops.yuv2h264(yuv_frame, save_file, (height, width), 30, 45, 0)
