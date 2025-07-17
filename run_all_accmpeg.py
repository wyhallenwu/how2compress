import os
import subprocess

q = 0.6
configs = {
    f"exp-accmpeg-1702-{q}q": "MOT17-02",
    f"exp-accmpeg-1704-{q}q": "MOT17-04",
    f"exp-accmpeg-1709-{q}q": "MOT17-09",
    f"exp-accmpeg-1710-{q}q": "MOT17-10",
    f"exp-accmpeg-1711-{q}q": "MOT17-11",
    f"exp-accmpeg-1713-{q}q": "MOT17-13",
}

# configs = os.listdir("/how2compress/configs")
# configs = [os.path.join("/how2compress/configs", config) for config in configs]

for k, v in configs.items():
    # print(k, v)
    subprocess.run(
        [
            "torchrun",
            "--nproc_per_node=3",
            "train_mb_det_accmpeg.py",
            "--name",
            k,
            "--dataset",
            v,
        ]
    )
