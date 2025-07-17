import os
from src.utils import image_ops
from tqdm import tqdm

# Hardcode all VisDrone validation sequences
DATASETS = [
    "uav0000086_00000_v",
    "uav0000117_02622_v",
    "uav0000137_00458_v",
    "uav0000182_00000_v",
    "uav0000268_05773_v",
    "uav0000305_00000_v",
    "uav0000339_00001_v"
]

# Setup paths
visdrone_root = "data/visdrone/VisDrone2019-VID-val"
yuv_root = "/how2compress/data/VisDroneYUV"
target = "/how2compress/data/VisDroneH264VAQ1"

if not os.path.exists(target):
    os.makedirs(target)

# Fixed resolution for VisDrone after resizing
width, height = 1920, 1080

for dataset in DATASETS:
    # Use the resized version of the sequence
    path = os.path.join(yuv_root, f"{dataset}_resized")
    yuv_frames = sorted(os.listdir(path))
    yuv_frames = [os.path.join(path, yuv_frame) for yuv_frame in yuv_frames]

    tgt_root = os.path.join(target, f"{dataset}_resized")
    if not os.path.exists(tgt_root):
        os.makedirs(tgt_root)

    for i, yuv_frame in tqdm(enumerate(yuv_frames), desc=f"Converting {dataset}"):
        save_file = os.path.join(tgt_root, f"{i+1:07d}.h264")  # Using 7 digits for VisDrone
        image_ops.yuv2h264(yuv_frame, save_file, (height, width), 30, 45, 1)  # Using same QP range as MOT 