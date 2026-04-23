"""Convert PNG folder to npz"""
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

img_dir = './error_dec/cifar/image'
files = sorted(os.listdir(img_dir))
print(f"Found {len(files)} files")

images = []
for f in tqdm(files, desc="Loading PNGs"):
    img = Image.open(os.path.join(img_dir, f))
    arr = np.array(img)
    if arr.ndim == 2:  # grayscale 혹시나 대비
        arr = np.stack([arr]*3, axis=-1)
    images.append(arr)

images = np.stack(images, axis=0)
print(f"Final shape: {images.shape}, dtype: {images.dtype}")
print(f"Value range: min={images.min()}, max={images.max()}")

np.savez('./iec_samples.npz', arr_0=images)
print("✓ Saved to ./iec_samples.npz")
