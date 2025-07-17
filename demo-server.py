import os
from fastapi import FastAPI, File, Form, UploadFile
import shutil
import json
import logging
from src.utils import image_ops, metrics
from src.dataset.dataloader import MOTDataset, collate_fn
from ultralytics import YOLO
import queue
from threading import Thread
import cv2
import torch
import supervision as sv
import numpy as np

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
q = queue.Queue()
mAP_all = []
comperssion_ratios = []
RESIZE_FACTOR=4

dataset = MOTDataset(
    dataset_dir="/data/MOT17Det/train",
    reference_dir="/data/detections",
    # ssim_label_dir="/how2compress/data/ssim_labels",
    yuv_dir="/data/MOT17DetYUV",
    resize_factor=RESIZE_FACTOR,
)
dataset.load_sequence("MOT17-02")

# Directory where the uploaded files will be stored
UPLOAD_DIRECTORY = "/data/received_videos"  # Replace with your directory path


inferencer = YOLO("/how2compress/pretrained/mot17-m.pt", verbose=False).to(
    "cuda"
)

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# Endpoint to upload files
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), metadata: str = Form(...)):
    try:
        # Parse the metadata from the client
        file_metadata = json.loads(metadata)
        file_name = file_metadata["filename"]
        unique_id = file_metadata["unique_id"]
        timestamp = file_metadata["timestamp"]
        start_idx = file_metadata["start_idx"]
        client_id = file_metadata["client_id"]
        compression_rate = file_metadata["compression_rate"]

        # Define the path to save the file
        file_path = os.path.join(UPLOAD_DIRECTORY, f"{client_id}-{file_name}")

        # logger.info(f"Received file: {file_name}, unique_id: {unique_id}, timestamp: {timestamp}")

        # Save the uploaded file to the directory
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        q.put((start_idx, file_path, compression_rate))

        
        return {
            "status": "success",
            "filename": file_name,
            "unique_id": unique_id,
            "timestamp": timestamp,
            "start_idx": start_idx,
            "client_id": client_id,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Example route to check the status of the server
@app.get("/status")
async def status():
    return {"status": "Server is running"}


def video_processing_worker():
      while True:
            try:
                kv = q.get()
                if kv is None:
                    continue
                start_idx, file_path, compression_rate = kv
                cap = cv2.VideoCapture(file_path)
                frames = []
                logger.info(f"Processing video: {file_path}")
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = image_ops.wrap_img(frame)
                    frame = image_ops.vit_transform_fn()(frame).to("cuda") 
                    frames.append(frame)
                cap.release()

                # logger.info(f"Number of frames: {len(frames)}")
                frames = torch.stack(frames)
                results = inferencer.predict(frames, classes=[1], verbose=False)
                results = metrics.yolo2sv(results)
                results = [
                    metrics.normalize_detections(
                        det,
                        (
                            dataset.curr_seq_property["width"],
                            dataset.curr_seq_property["height"],
                        ),
                    )
                    for det in results
                ]
                labels = dataset.curr_labels[start_idx-1: start_idx+24]
                mAP = sv.MeanAveragePrecision.from_detections(results, labels)
                mAP_all.append(mAP.map50)
                comperssion_ratios.append(compression_rate)
                # logger.info(f"mAP: {round(mAP.map50/0.84 * 100, 1)}%")
                logger.info(f"average mAP50 across this video stream: {round(np.mean(mAP_all)/0.84 * 100, 3)}%")
                logger.info(f"compression rate: {compression_rate}%")
                # logger.info(f"average compression rate across this video stream: {np.mean(comperssion_ratios)}%")
            except Exception as e:
                # logger.error(f"Error processing video: {file_path}")
                # logger.error(str(e))
                pass

def start_video_processing_worker():
    worker_thread = Thread(target=video_processing_worker)
    # worker_thread.daemon = True
    worker_thread.start()

if __name__ == "__main__":
    start_video_processing_worker()
    logger.info("start model completion")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=60)
    
  
            
