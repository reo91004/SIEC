"""Create CIFAR-10 reference npz for evaluator_FID.py"""
import numpy as np
from torchvision.datasets import CIFAR10

print("Downloading CIFAR-10 if needed...")
dataset = CIFAR10(root='./data_cifar10', train=True, download=True)

print(f"Total images: {len(dataset)}")
images = []
for i, (img, _) in enumerate(dataset):
    arr = np.array(img)
    images.append(arr)
    if (i+1) % 10000 == 0:
        print(f"  Processed {i+1}/{len(dataset)}")

images = np.stack(images, axis=0)
print(f"Final shape: {images.shape}, dtype: {images.dtype}")
print(f"Value range: min={images.min()}, max={images.max()}")

np.savez('./cifar10_reference.npz', arr_0=images)
print("✓ Saved to ./cifar10_reference.npz")
