import os
import ast
import numpy as np
import csv
import seaborn as sns
import matplotlib.pyplot as plt

filename = "/how2compress/pretrained/train/exp1704-1/val_log-0.993-0.978.txt"
line = None
with open(filename, "r") as f:
    line = f.readlines()[-3]
    reader = csv.reader([line.strip()], delimiter=",", quotechar='"')
    line = next(reader)
    line = line[-1]
    # print(line)

decisions = ast.literal_eval(f"{line}")
print(len(decisions))
decisions = np.array(decisions).reshape(68, 120)
print(decisions.shape)
plt.figure(figsize=(120, 90))
sns.heatmap(decisions, cmap="coolwarm", cbar=True, annot=True, fmt=".2f")
plt.tight_layout()
plt.savefig("graph/decisions.pdf")
