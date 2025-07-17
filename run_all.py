import os
import subprocess

configs = os.listdir("/how2compress/configs")
configs = [os.path.join("/how2compress/configs", config) for config in configs]

for config in configs:
    subprocess.run(
        ["torchrun", "--nproc_per_node=3", "train_mb_det.py", "--config", config]
    )
