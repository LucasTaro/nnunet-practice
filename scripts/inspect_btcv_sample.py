
from pathlib import Path
from io import BytesIO
import zipfile

import numpy as np
from PIL import Image

zip_path = Path("/mnt/gptdata/maoyutao/data/btcv.zip")

image_name = "btcv/Training/image/img0001/63.jpg"
mask_name = "btcv/Training/mask/img0001/63.npy"

with zipfile.ZipFile(zip_path, "r") as z:

    # 读取 JPG 图像
    with Image.open(BytesIO(z.read(image_name))) as im:
        print("Image mode:", im.mode)
        image = np.array(im)

    # 读取 NPY 分割标签
    mask = np.load(
        BytesIO(z.read(mask_name)),
        allow_pickle=False
    )

print("\n===== CT Image =====")
print("Shape:", image.shape)
print("Dtype:", image.dtype)
print("Min:", image.min())
print("Max:", image.max())

print("\n===== Segmentation Mask =====")
print("Shape:", mask.shape)
print("Dtype:", mask.dtype)
print("Min:", mask.min())
print("Max:", mask.max())
print("Unique labels:", np.unique(mask))

# 检查空间尺寸
assert image.shape[:2] == mask.shape[:2], \
    "Image and mask size mismatch!"

print("\nImage and mask spatial sizes match!")
