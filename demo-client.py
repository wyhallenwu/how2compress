import torch
import cv2
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model.hgmodel import MobileVitV2
from src.model.utils import ems2selections
from src.model.am_model import AccMpeg
from src.utils import image_ops, load, cals, metrics
from src.dataset.dataloader import MOTDataset, collate_fn
import os
from ultralytics import YOLO
from tqdm import tqdm
import supervision as sv
import time
import argparse
import subprocess
import numpy as np
import tempfile
import shutil
import aiohttp
import asyncio
import json
from uuid import uuid4

async def async_send_video(session, file_path, server_addr, start_idx, client_id, compression_rate):
    unique_id = uuid4()
    timestamp = int(time.time())

    file_metadata = {
        "filename": os.path.basename(file_path),
        "timestamp": timestamp,
        "unique_id": str(unique_id),
        "start_idx": start_idx,
        "client_id": client_id,
        "compression_rate": compression_rate,
    }

    with open(file_path, "rb") as f:
        data = aiohttp.FormData()
        data.add_field("file", f, filename=file_path)
        data.add_field("metadata", json.dumps(file_metadata))

        async with session.post(server_addr, data=data) as response:
            if response.status == 200:
                print(f"Successfully sent {file_path}")
            else:
                print(f"Failed to send {file_path}")

async def main(server_addr, client_id):
    # mapping = {0: 45, 1: 43, 2: 37, 3: 34, 4: 30}
    mapping = {0: 45, 1: 43, 2: 37, 3: 34, 4: 30}
    dataset = "MOT17-02"
    src_root = f"/data/MOT17YUVCHUNK25/{dataset}/"
    videos = sorted(os.listdir(src_root))
    # num_videos = len(videos)
    # accmpeg_model = (
    #     # "/how2compress/pretrained/train/exp-accmpeg-1702-0.35q/1-0.0002956777562219681.pth"
    #     # "/how2compress/pretrained/train/exp-accmpeg-1704-0.35q/1--0.007357695757508331.pth"
    #     # "/how2compress/pretrained/train/exp-accmpeg-1709-0.45q/0--0.0059172190316897355.pth"
    #     # "/how2compress/pretrained/train/exp-accmpeg-1710-0.45q/0--0.009722388405215221.pth"
    #     # "/how2compress/pretrained/train/exp-accmpeg-1711-0.45q/0--0.011230770333203854.pth"
    #     "/how2compress/pretrained/train/exp-accmpeg-1713-0.45q/0--0.03347167812188889.pth"
    # )

    # ours_model = "/how2compress/pretrained/train/exp1702-1-1/1-0.4764635543758309+-0.001443803332996929-0.988-0.921.pth"
    # ours_model = "/how2compress/pretrained/train/exp1704-1/1-0.5303217708720351+-0.0042760108615707-0.993-0.978.pth"
    # ours_model = "/how2compress/pretrained/train/exp1709-1/1-0.5740472400763013+-0.007589454466788492-0.982-0.931.pth"
    # ours_model = "/how2compress/pretrained/train/exp1710-1/1-0.4620776571761211+-0.0023735605196651965-0.993-0.971.pth"
    # ours_model = "/how2compress/pretrained/train/exp1711-1-1/1-0.5398975733473949+0.0023943903714293002-0.991-0.947.pth"
    ours_model = "/how2compress/pretrained/train/exp1713-1-1/1-0.3470107100380082+-0.01633170276732193-0.991-0.952.pth"

    root_dir = f"video-result/demo/{dataset}"
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    raw = f"/data/MOT17Det/train/{dataset}/img1"

    num = 25
    bf = 3
    num_frames = len(os.listdir(raw))
    # start_idx = 1

    DEVICE = "cuda"
    RESIZE_FACTOR = 4
    BATCH_SIZE = 1

    mb_w, mb_h = cals.macroblocks_wh(1920, 1080)
    transform = image_ops.vit_transform_fn()
    resizer = image_ops.resize_img_tensor((mb_h * 4, mb_w * 4))

    model2 = MobileVitV2()
    model2.load_state_dict(torch.load(ours_model))
    model2.to(DEVICE)
    model2.set_output_size((mb_h, mb_w))


    async with aiohttp.ClientSession() as session:

        for idx, video in tqdm(enumerate(videos)):
            start_idx = 1 + idx * num
            video = os.path.join(src_root, video)

            with tempfile.TemporaryDirectory() as tmpdirname:
                for j in range(num):
                    idx1 = start_idx + j
                    if idx1 > num_frames:
                        break
                    shutil.copyfile(
                        os.path.join(raw, f"{idx1:06d}.jpg"),
                        os.path.join(tmpdirname, f"{j:06d}.jpg"),
                    )

                # raw
                with open("/myh264/operation_mode_file", "w") as f:
                    f.write("0,30,")
                
                subprocess.run(
                    [
                        "/myh264/bin/ffmpeg",
                        "-y",
                        "-i",
                        f"{tmpdirname}/%06d.jpg",
                        "-start_number",
                        str(0),
                        "-vframes",
                        str(num),
                        "-framerate",
                        "25",
                        "-qp",
                        "10",
                        "-pix_fmt",
                        "yuv420p",
                        os.path.join(root_dir, f"h264-raw-{(idx+1):02d}.mp4"),
                    ]
                )

                raw_size = os.path.getsize(os.path.join(root_dir, f"h264-raw-{(idx+1):02d}.mp4"))


                with open("/myh264/operation_mode_file", "w") as f:
                    f.write("20,")

                # ours
                with open("/myh264/qp_matrix_file", "w") as f:
                    for i in range(num):
                        # count += 1
                        image = cv2.imread(os.path.join(raw, f"{(i+start_idx):06d}.jpg"))
                        image = image_ops.wrap_img(image)
                        image = transform(image).unsqueeze(0).to(DEVICE)
                        # print(image.shape)
                        resize_image = resizer(image)
                        ems_map_indices, ems_map_v, selections = model2(resize_image)
                        # print(selections[0])
                        selections = [mapping[i] for _, i in selections[0]]

                        values, counts = np.unique(selections, return_counts=True)
                        print(f"h264 - values: {values}, counts: {counts}")

                        matrix = np.reshape(selections, (mb_h, mb_w))
                        for row in matrix:
                            f.write(" ".join(map(str, row)) + "\n")

                subprocess.run(
                    [
                        "/myh264/bin/ffmpeg",
                        "-y",
                        "-i",
                        f"{tmpdirname}/%06d.jpg",
                        "-start_number",
                        str(0),
                        "-vframes",
                        str(num),
                        "-framerate",
                        "25",
                        "-qp",
                        "10",
                        "-pix_fmt",
                        "yuv420p",
                        os.path.join(root_dir, f"h264-ours-{(idx+1):02d}.mp4"),
                    ]
                )
                compressed_size = os.path.getsize(os.path.join(root_dir, f"h264-ours-{(idx+1):02d}.mp4"))
                compression_rate = round(1 - compressed_size / raw_size, 3) * 100
                video_file = os.path.join(root_dir, f"h264-ours-{(idx+1):02d}.mp4")
                await async_send_video(session, video_file, server_addr, start_idx, client_id, compression_rate)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--server-addr", type=str, default="http://localhost:8000/upload")
    args = parser.parse_args()
    asyncio.run(main(args.server_addr, args.id))