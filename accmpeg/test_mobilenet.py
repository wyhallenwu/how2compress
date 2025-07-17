from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests
import torch.nn as nn

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open(
    "/how2compress/data/MOTCOCO/images/train-MOT17-02/000001.jpg"
)
print(image.size)

preprocessor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
model = AutoModelForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")
model.classifier = nn.Linear(1280, 2, bias=True)
print(model)

inputs = preprocessor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits
print(logits.shape)

# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
