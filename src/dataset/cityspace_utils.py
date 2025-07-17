import torchvision

dataset = torchvision.datasets.Cityscapes(
    "/how2compress/cityspace", split="train", mode="fine"
)
img, smnt = dataset[0]
print(type(img), type(smnt))
