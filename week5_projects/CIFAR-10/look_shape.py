import pickle
import numpy as np
from PIL import Image 

path="week5_projects/CIFAR-10/cifar-10-python/cifar-10-batches-py/data_batch_1"

with open(path,"rb") as f:
    batch = pickle.load(f,encoding="bytes")

data = batch[b"data"]
lables = batch[b"labels"]

print(data.shape)
print(len(lables))
image = data[0].reshape(3, 32, 32).transpose(1, 2, 0)
image = Image.fromarray(image)
image.show()