./ffmpeg-7.1.1/ffmpeg -benchmark -framerate 30 -i data/17-person-0.6/images/train-MOT17-02/%06d.jpg -c:v libvpx-vp9 -crf 30 -b:v 0 -deadline realtime -threads 1 further_results/mot1702-vp9.webm
./ffmpeg-7.1.1/ffmpeg -benchmark -framerate 30 -i data/17-person-0.6/images/train-MOT17-02/%06d.jpg -c:v libx265 -tune zerolatency -preset veryfast -crf 30 -x265-params "pools=1" -threads 1 further_results/mot1702-xh265.mp4
./ffmpeg-7.1.1/ffmpeg -benchmark -framerate 30 -i data/17-person-0.6/images/train-MOT17-02/%06d.jpg -c:v libvvenc -preset 0 -qp 30  -pix_fmt yuv420p -threads 1 further_results/mot1702-vvc.mp4
./ffmpeg-7.1.1/ffmpeg -benchmark -framerate 30 -i data/17-person-0.6/images/train-MOT17-02/%06d.jpg -c:v libaom-av1 -crf 30 -b:v 0 -cpu-used 1 -usage realtime -row-mt 1 -threads 1 -g 30 -pix_fmt yuv420p further_results/mot1702-av1.mkv


./ffmpeg-7.1.1/ffmpeg -benchmark -framerate 30 -i data/17-person-0.6/images/train-MOT17-04/%06d.jpg -c:v libvpx-vp9 -crf 30 -b:v 0 -deadline realtime -threads 1 further_results/mot1704-vp9.webm
./ffmpeg-7.1.1/ffmpeg -benchmark -framerate 30 -i data/17-person-0.6/images/train-MOT17-04/%06d.jpg -c:v libx265 -tune zerolatency -preset veryfast -crf 30 -x265-params "pools=1" -threads 1 further_results/mot1704-xh265.mp4
./ffmpeg-7.1.1/ffmpeg -benchmark -framerate 30 -i data/17-person-0.6/images/train-MOT17-04/%06d.jpg -c:v libvvenc -preset 0 -qp 30  -pix_fmt yuv420p -threads 1 further_results/mot1704-vvc.mp4
./ffmpeg-7.1.1/ffmpeg -benchmark -framerate 30 -i data/17-person-0.6/images/train-MOT17-04/%06d.jpg -c:v libaom-av1 -crf 30 -b:v 0 -cpu-used 1 -usage realtime -row-mt 1 -threads 1 -g 30 -pix_fmt yuv420p further_results/mot1704-av1.mkv


./ffmpeg-7.1.1/ffmpeg -benchmark -framerate 30 -i data/17-person-0.6/images/train-MOT17-09/%06d.jpg -c:v libvpx-vp9 -crf 30 -b:v 0 -deadline realtime -threads 1 further_results/mot1709-vp9.webm
./ffmpeg-7.1.1/ffmpeg -benchmark -framerate 30 -i data/17-person-0.6/images/train-MOT17-09/%06d.jpg -c:v libx265 -tune zerolatency -preset veryfast -crf 30 -x265-params "pools=1" -threads 1 further_results/mot1709-xh265.mp4
./ffmpeg-7.1.1/ffmpeg -benchmark -framerate 30 -i data/17-person-0.6/images/train-MOT17-09/%06d.jpg -c:v libvvenc -preset 0 -qp 30  -pix_fmt yuv420p -threads 1 further_results/mot1709-vvc.mp4
./ffmpeg-7.1.1/ffmpeg -benchmark -framerate 30 -i data/17-person-0.6/images/train-MOT17-09/%06d.jpg -c:v libaom-av1 -crf 30 -b:v 0 -cpu-used 1 -usage realtime -row-mt 1 -threads 1 -g 30 -pix_fmt yuv420p further_results/mot1709-av1.mkv

./ffmpeg-7.1.1/ffmpeg -benchmark -framerate 30 -i data/17-person-0.6/images/train-MOT17-10/%06d.jpg -c:v libvpx-vp9 -crf 30 -b:v 0 -deadline realtime -threads 1 further_results/mot1710-vp9.webm
./ffmpeg-7.1.1/ffmpeg -benchmark -framerate 30 -i data/17-person-0.6/images/train-MOT17-10/%06d.jpg -c:v libx265 -tune zerolatency -preset veryfast -crf 30 -x265-params "pools=1" -threads 1 further_results/mot1710-xh265.mp4
./ffmpeg-7.1.1/ffmpeg -benchmark -framerate 30 -i data/17-person-0.6/images/train-MOT17-10/%06d.jpg -c:v libvvenc -preset 0 -qp 30  -pix_fmt yuv420p -threads 1 further_results/mot1710-vvc.mp4
./ffmpeg-7.1.1/ffmpeg -benchmark -framerate 30 -i data/17-person-0.6/images/train-MOT17-10/%06d.jpg -c:v libaom-av1 -crf 30 -b:v 0 -cpu-used 1 -usage realtime -row-mt 1 -threads 1 -g 30 -pix_fmt yuv420p further_results/mot1710-av1.mkv

./ffmpeg-7.1.1/ffmpeg -benchmark -framerate 30 -i data/17-person-0.6/images/train-MOT17-11/%06d.jpg -c:v libvpx-vp9 -crf 30 -b:v 0 -deadline realtime -threads 1 further_results/mot1711-vp9.webm
./ffmpeg-7.1.1/ffmpeg -benchmark -framerate 30 -i data/17-person-0.6/images/train-MOT17-11/%06d.jpg -c:v libx265 -tune zerolatency -preset veryfast -crf 30 -x265-params "pools=1" -threads 1 further_results/mot1711-xh265.mp4
./ffmpeg-7.1.1/ffmpeg -benchmark -framerate 30 -i data/17-person-0.6/images/train-MOT17-11/%06d.jpg -c:v libvvenc -preset 0 -qp 30  -pix_fmt yuv420p -threads 1 further_results/mot1711-vvc.mp4
./ffmpeg-7.1.1/ffmpeg -benchmark -framerate 30 -i data/17-person-0.6/images/train-MOT17-11/%06d.jpg -c:v libaom-av1 -crf 30 -b:v 0 -cpu-used 1 -usage realtime -row-mt 1 -threads 1 -g 30 -pix_fmt yuv420p further_results/mot1711-av1.mkv


./ffmpeg-7.1.1/ffmpeg -benchmark -framerate 30 -i data/17-person-0.6/images/train-MOT17-13/%06d.jpg -c:v libvpx-vp9 -crf 30 -b:v 0 -deadline realtime -threads 1 further_results/mot1713-vp9.webm
./ffmpeg-7.1.1/ffmpeg -benchmark -framerate 30 -i data/17-person-0.6/images/train-MOT17-13/%06d.jpg -c:v libx265 -tune zerolatency -preset veryfast -crf 30 -x265-params "pools=1" -threads 1 further_results/mot1713-xh265.mp4
./ffmpeg-7.1.1/ffmpeg -benchmark -framerate 30 -i data/17-person-0.6/images/train-MOT17-13/%06d.jpg -c:v libvvenc -preset 0 -qp 30  -pix_fmt yuv420p -threads 1 further_results/mot1713-vvc.mp4
./ffmpeg-7.1.1/ffmpeg -benchmark -framerate 30 -i data/17-person-0.6/images/train-MOT17-13/%06d.jpg -c:v libaom-av1 -crf 30 -b:v 0 -cpu-used 1 -usage realtime -row-mt 1 -threads 1 -g 30 -pix_fmt yuv420p further_results/mot1713-av1.mkv