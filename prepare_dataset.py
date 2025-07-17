import os
import tempfile
from src.dataset import mot_utils, process
from tqdm import tqdm
from src.utils.cals import make_even
import cv2
# AVAILABLE_DATASETS = {
#     "MOT17-02": "MOT17-02-DetUniQP",
#     "MOT17-04": "MOT17-04-DetUniQP",
#     "MOT17-05": "MOT17-05-DetUniQP",
#     "MOT17-09": "MOT17-09-DetUniQP",
#     "MOT17-10": "MOT17-10-DetUniQP",
#     "MOT17-11": "MOT17-11-DetUniQP",
#     "MOT17-13": "MOT17-13-DetUniQP",
#     "MOT20-01": "MOT20-01-DetUniQP",
#     "MOT20-02": "MOT20-02-DetUniQP",
#     "MOT20-03": "MOT20-03-DetUniQP",
#     "MOT20-05": "MOT20-05-DetUniQP",
# }

# CONST_QP = [30, 34, 37, 41, 45]
# CONST_QP = [25, 30, 31]
CONST_QP = [30]
FRAME_PER_CHUNK = 1


def prepare_MOT(
    src_dir: str,
    yuv_dest_dir: str,
    h264_dest_dir: str,
    dec_dest_dir: str,
    frames_per_chunk: int,
):
    train_root = os.path.join(src_dir, "train")
    sequences = sorted(os.listdir(train_root))
    for sequence in tqdm(sequences, desc="prepare_MOT_yuv"):
        sequence_path = os.path.join(train_root, sequence)
        seq_info = mot_utils.parse_seqinfo(sequence_path)
        source_folder = os.path.join(sequence_path, "img1")

        # convert frames to YUV
        yuv_dest_folder = os.path.join(yuv_dest_dir, sequence)
        if not os.path.exists(yuv_dest_folder):
            os.makedirs(yuv_dest_folder)

        process.convert2yuv(source_folder, yuv_dest_folder, frames_per_chunk)

        # for base_qp in CONST_QP:
        #     # convert YUV to h264 chunk with uniform QP
        #     h264_dest_folder = os.path.join(h264_dest_dir, sequence, str(base_qp))
        #     if not os.path.exists(h264_dest_folder):
        #         os.makedirs(h264_dest_folder)

        #     process.encode_chunks(
        #         yuv_dest_folder,
        #         h264_dest_folder,
        #         make_even(int(seq_info["imWidth"])),
        #         make_even(int(seq_info["imHeight"])),
        #         base_qp,
        #         base_qp,
        #         None,
        #     )

        # # decode the raw h264 to jpg
        # dec_dest_folder = os.path.join(dec_dest_dir, AVAILABLE_DATASETS[sequence], str(base_qp))
        # if not os.path.exists(dec_dest_folder):
        #     os.makedirs(dec_dest_folder)

        # process.decode_chunks(h264_dest_folder, dec_dest_folder)


def prepare_panda(
    src_dir: str,
    yuv_dir: str,
    h264_dest_dir: str,
    frames_per_chunk: int,
):
    # read 2K frames
    sequences = sorted(os.listdir(src_dir))
    for sequence in tqdm(sequences, desc="prepare_panda_yuv"):
        sequence_path = os.path.join(src_dir, sequence)
        source_folder = os.path.join(sequence_path, "2560x1440")

        # convert frames to YUV
        yuv_dest_folder = os.path.join(yuv_dir, sequence, "2560x1440")
        if not os.path.exists(yuv_dest_folder):
            os.makedirs(yuv_dest_folder)

        process.convert2yuv(source_folder, yuv_dest_folder, frames_per_chunk)

        for base_qp in CONST_QP:
            # convert YUV to h264 chunk with uniform QP
            h264_dest_folder = os.path.join(h264_dest_dir, sequence, str(base_qp))
            if not os.path.exists(h264_dest_folder):
                os.makedirs(h264_dest_folder)

            process.encode_chunks(
                yuv_dest_folder,
                h264_dest_folder,
                make_even(2560),
                make_even(1440),
                base_qp,
                base_qp,
                None,
            )

            # decode the raw h264 to jpg
            # dec_dest_folder = os.path.join(dec_dest_dir, sequence, str(base_qp))
            # if not os.path.exists(dec_dest_folder):
            #     os.makedirs(dec_dest_folder)

            # process.decode_chunks(h264_dest_folder, dec_dest_folder)
    pass


def prepare_aicity(src: str, yuv_tgt: str, h264_tgt: str, frames_per_chunk: int):
    sequences = sorted(os.listdir(src))
    for sequence in tqdm(sequences, desc="prepare_aicity_yuv"):
        sequence_path = os.path.join(src, sequence)
        positions = sorted(os.listdir(os.path.join(src, sequence)))
        for pos in positions:
            sequence_path = os.path.join(sequence_path, pos, "vdo.avi")

            yuv_dest_folder = os.path.join(yuv_tgt, sequence, pos)
            if not os.path.exists(yuv_dest_folder):
                os.makedirs(yuv_dest_folder)

            with tempfile.TemporaryDirectory() as tempdir:
                frame_count = 1
                cap = cv2.VideoCapture(sequence_path)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cv2.imwrite(os.path.join(tempdir, f"{frame_count:06d}.jpg"), frame)
                    frame_count += 1
                cap.release()
                process.convert2yuv(tempdir, yuv_dest_folder, frames_per_chunk)
            for base_qp in CONST_QP:
                h264_dest_folder = os.path.join(h264_tgt, sequence, pos, str(base_qp))
                if not os.path.exists(h264_dest_folder):
                    os.makedirs(h264_dest_folder)
                process.encode_chunks(
                    yuv_dest_folder,
                    h264_dest_folder,
                    make_even(1080),
                    make_even(1920),
                    base_qp,
                    base_qp,
                    None,
                )


def prepare_VisDrone(
    src_dir: str,
    yuv_dest_dir: str,
    h264_dest_dir: str,
    dec_dest_dir: str,
    frames_per_chunk: int,
):
    """Prepare VisDrone dataset by converting frames to YUV, encoding to h264, and decoding back to jpg.
    
    Args:
        src_dir (str): Source directory containing VisDrone sequences
        yuv_dest_dir (str): Destination directory for YUV files
        h264_dest_dir (str): Destination directory for h264 files
        dec_dest_dir (str): Destination directory for decoded jpg files
        frames_per_chunk (int): Number of frames per chunk
    """
    # Get all sequences and filter for resized folders
    sequences = [seq for seq in sorted(os.listdir(src_dir)) if "resized" in seq]
    for sequence in tqdm(sequences, desc="prepare_VisDrone_yuv"):
        sequence_path = os.path.join(src_dir, sequence)
        source_folder = sequence_path  # Direct jpg files in sequence folder
        print(f"source_folder: {source_folder}")

        # convert frames to YUV using resized frames
        yuv_dest_folder = os.path.join(yuv_dest_dir, sequence)
        if not os.path.exists(yuv_dest_folder):
            os.makedirs(yuv_dest_folder)
        print(f"yuv_dest_folder: {yuv_dest_folder}")

        process.convert2yuv(source_folder, yuv_dest_folder, frames_per_chunk, num_zeros=7)

        for base_qp in CONST_QP:
            # convert YUV to h264 chunk with uniform QP
            h264_dest_folder = os.path.join(h264_dest_dir, sequence, str(base_qp))
            if not os.path.exists(h264_dest_folder):
                os.makedirs(h264_dest_folder)

            process.encode_chunks(
                yuv_dest_folder,
                h264_dest_folder,
                make_even(1920),
                make_even(1080),
                base_qp,
                base_qp,
                None,
            )

            # decode the raw h264 to jpg
            dec_dest_folder = os.path.join(dec_dest_dir, sequence, str(base_qp))
            if not os.path.exists(dec_dest_folder):
                os.makedirs(dec_dest_folder)

            process.decode_chunks(h264_dest_folder, dec_dest_folder)


if __name__ == "__main__":
    # prepare_MOT(
    #     "/how2compress/data/MOT17Det",
    #     "/how2compress/data/MOT17YUVCHUNK25",
    #     "/how2compress/data/val-bitrate",
    #     "/how2compress/data/val-bitrate",
    #     FRAME_PER_CHUNK,
    # )
    # prepare_MOT(
    #     "/how2compress/data/MOT20Det",
    #     "/how2compress/data/MOT20DetYUV",
    #     "/how2compress/data/detections",
    #     "/how2compress/data/detections",
    #     FRAME_PER_CHUNK,
    # )
    # prepare_panda(
    #     "/how2compress/data/pandasRS",
    #     "/how2compress/data/pandasYUV",
    #     "/how2compress/data/pandasH264",
    #     1,
    # )

    # prepare_aicity(
    #     "/how2compress/data/aicity/train",
    #     "/how2compress/data/aicityYUV",
    #     "/how2compress/data/aicityH264",
    #     FRAME_PER_CHUNK,
    # )

    prepare_VisDrone(
        "/how2compress/data/visdrone/VisDrone2019-VID-val/sequences",
        "/how2compress/data/VisDroneYUV",
        "/how2compress/data/VisDroneH264",
        "/how2compress/data/VisDroneDecoded",
        FRAME_PER_CHUNK,
    )
