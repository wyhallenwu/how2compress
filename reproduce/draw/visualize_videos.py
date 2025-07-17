import cv2
import os

# Sequence ID
seq = "02"

# Define video filenames
video_files = {
    "uni": f"video-uni30-MOT17-{seq}.mp4",
    "aq": f"mot17{seq}-ap0.mp4",
    "accmpeg": f"video-accmpeg-MOT17-{seq}.mp4",
    "ours": f"video-ours-MOT17-{seq}.mp4"
}

# Directory where videos are stored
video_dir = "video-result"

# Directory to save the first frame images
output_dir = "first_frames"
os.makedirs(output_dir, exist_ok=True)

for label, filename in video_files.items():
    video_path = os.path.join(video_dir, filename)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was successfully opened
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        continue

    # Read the first frame
    ret, frame = cap.read()

    if ret:
        # Construct output image path
        output_image_path = os.path.join(output_dir, f"{label}_MOT17-{seq}_first_frame.jpg")
        cv2.imwrite(output_image_path, frame)
        print(f"First frame saved: {output_image_path}")
    else:
        print(f"Error: Failed to read the first frame from {video_path}")

    # Release the video capture object
    cap.release()

