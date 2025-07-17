import cv2

video_accmpeg = "/how2compress/video-result/MOT17-13/h264-accmpeg-01.mp4"
video_ours = "/how2compress/video-result/MOT17-13/h264-ours-01.mp4"
videos_raw = "/how2compress/data/UNI25CHUNK/MOT17-13/uni-01.mp4"

cap_accmpeg = cv2.VideoCapture(video_accmpeg)
while True:
    ret, frame_accmpeg = cap_accmpeg.read()
    if not ret:
        break
    cv2.imwrite("graph/frame_accmpeg.png", frame_accmpeg)
    break
cap_accmpeg.release()

cap_ours = cv2.VideoCapture(video_ours)
while True:
    ret, frame_ours = cap_ours.read()
    if not ret:
        break
    cv2.imwrite("graph/frame_ours.png", frame_ours)
    break
cap_ours.release()

cap_raw = cv2.VideoCapture(videos_raw)
while True:
    ret, frame_raw = cap_raw.read()
    if not ret:
        break
    cv2.imwrite("graph/frame_raw.png", frame_raw)
    break
cap_raw.release()
