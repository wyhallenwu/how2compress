
<div align="center">
<h1>How2Compress: Efficient Edge Video Analytics via Adaptive Granular Video Compression</h1>


**[CDSN Lab, KAIST](https://cds.kaist.ac.kr/)**

Yuheng Wu, Thanh-Tung Nguyen, Lucas Liebe, Nhat-Quang Tau, Pablo Espinosa Campos, Jinghan Cheng, Dongman Lee
</div>

```bibtex
@inproceedings{wu2025how2compress,
  title={How2Compress: Efficient Edge Video Analytics via Adaptive Granular Video Compression},
  author={Wu, Yuheng and Nguyen, Thanh-Tung and Lucas Liebe and  Tau, Nhat-Quang and Pablo Espinosa Campos and Cheng, Jinghan and Lee, Dongman},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  year={2025}
}
```

This repository contains the official implementation of the paper:
"How2Compress: Efficient Edge Video Analytics via Adaptive Granular Video Compression"

## Env Setup
We provide two options for setting up the environment: using a pre-built Docker image or building from scratch.

### [option1] Docker Image (Recommended)
We provide a Docker image to facilitate quick reproduction of our experiments and results. The Docker image is available at the following link: https://hub.docker.com/r/wuyuheng/how2compress

We strongly recommend using the Docker image to avoid the complexity of manual environment configuration. Some codes such as Nvidia Video SDK, please refer to the implementaion.

### [option2] Build From Scratch
1. Python Environment Setup
```bash
# Using pip
pip install -r how2compress-requirements.txt

# Or using conda
conda env create -f how2compress-env.yaml
```

2. Install Advanced Video Codecs
```bash
wget https://ffmpeg.org/releases/ffmpeg-7.1.1.tar.xz
tar -xf ffmpeg-7.1.1.tar.xz

# compile with x265, vp9, vvc, av1 that used in our paper, make sure you install corresponding codec beforehand
./configure \
  --enable-shared \
  --enable-libx264 \
  --enable-libx265 \
  --enable-libmp3lame \
  --enable-libopus \
  --enable-libvpx \
  --enable-libaom \       
  --enable-libvvenc \        
  --enable-gpl \
  --enable-nonfree \
  --disable-x86asm

make -j$(nproc)
sudo make install
```

Ensure you have installed all required codec libraries (e.g., libx265, libvpx, libaom, etc.) before building.

3. Build the Modified H.264 Codec

Navigate to the myh264 directory and execute:
```bash
cd myh264/
bash build.sh
```

If you encounter errors related to missing .so files, set the appropriate library paths as follows (replace paths with actual locations):
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/myh264/ffmpeg-3.4.8/libavcodec:/myh264/ffmpeg-3.4.8/libavdevice/:/myh264/ffmpeg-3.4.8/libavfilter:/myh264/ffmpeg-3.4.8/libavformat:/myh264/ffmpeg-3.4.8/libavresampler:/myh264/ffmpeg-3.4.8/libavutil:/myh264/ffmpeg-3.4.8/libpostproc:/myh264/ffmpeg-3.4.8/libswresample:/myh264/ffmpeg-3.4.8/libswscale:/myh264/x264
export PATH=$PATH:/myh264/x264
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/myh264/x264
```

4. Compile NVENC for Fast Training-Side Encoding
```bash
cmake .. \
  -DCUVID_LIB=/how2compress/nvenc_codec_sdk_11.0.10/Lib/linux/stubs/x86_64/libnvcuvid.so \
  -DNVENCODEAPI_LIB=/how2compress/nvenc_codec_sdk_11.0.10/Lib/linux/stubs/x86_64/libnvidia-encode.so \
  -DCMAKE_INCLUDE_PATH=/how2compress/nvenc_codec_sdk_11.0.10/Interface
```
> Note: You must comply with NVIDIA's Video Codec SDK licensing terms. Our implementation modifies NVENC to support the _emphasis map feature_ as in https://docs.nvidia.com/video-technologies/video-codec-sdk/11.1/nvenc-video-encoder-api-prog-guide/index.html#emphasis-map


## Train

> Before training, we pre-encode and convert the frames into video chunks. Please refer to `prepare_dataset/` and convert data format by yourself. Due to the large volumne of dataset and anonymous requirement, it's hard to find an appropriate way to share, we will release later via huggingface dataset.

All training parameters are configurable in the config/ directory.

How2compress (Ours):
```bash
# MOT dataset
torchrun --nproc_per_node=3 train_mb_det.py --config <configs/mot1702.yaml>

# AICITY dataset
torchrun --nproc_per_node=3 train_mb_det_aicity.py --config <configs/aicity.yaml>
```

To reproduce baselines:

Where2Compress (ie, AccMPEG):
```bash
# tune parameter in each file here

# MOT dataset 
torchrun --nproc_per_node=3 train_mb_det_accmpeg.py 

# AICITY dataset
torchrun --nproc_per_node=3 train_mb_det_accmpeg_aicity.py
```

For When2Compress (CASVA, ILCAS): Since our focus is on quality adjustment (not resolution or frame rate), we perform an exhaustive search of frame-level QP values in the range [28, 35].


## Reproducing Our Results

### Evaluation
To reproduce results from our paper:

1. After training, ensure all checkpoints are saved in the `pretrained/` directory.

2. Use the scripts under `eval/` to re-run the experiments that produce each table and figure in the paper.

3. After evaluation, you will obtain all compressed video outputs categorized accordingly.

4. Use the script `reproduce/compute_result` to compute the statistics used for validation.

### Visualization
To reproduce the figures from the paper, refer to the scripts in `reproduce/draw/`, which contain all relevant figure generation source code.


## Tips
The current codebase is somewhat disorganized, but we are working to clean and update it as soon as possible. To reproduce our results, follow these general steps:

**Data Preparation:** Convert your video data or frames into the YUV format. Ensure that the frame width and height are divisible by 16 to maintain compatibility with standard codecs. You can refer to the `prepare_dataset.py` script (files with this prefix) for guidance on YUV generation.

**Script Integration:** Once the YUV files are ready, align the video paths with the training script accordingly.

Note: Due to the inherent non-determinism of our algorithm, the compression rate you obtain may not exactly match the results reported in our paper. However, the method consistently outperforms coarse-grained baselines across multiple runs. We observed minor variations in compression rate across trials, which is expected.

In some cases, the naive codec Adaptive Quantization (AQ) may outperform our method, as reported in the main table of our paper. This is primarily because naive AQ can generate more skip-mode macroblocks (MBs), which are highly efficient for compression.

## Acknowledgements
We would like to express our sincere gratitude to the authors of [AccMPEG](https://arxiv.org/abs/2204.12534) for their outstanding work. Our research builds significantly upon their open-source contributions, without which this work would not have been possible.
> Please refer to AccMPEG at: https://github.com/KuntaiDu/AccMPEG
