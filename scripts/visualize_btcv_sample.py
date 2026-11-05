
from pathlib import Path
from io import BytesIO
import zipfile

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

zip_path = Path("/mnt/gptdata/maoyutao/data/btcv.zip")

image_name = "btcv/Training/image/img0001/63.jpg"
mask_name = "btcv/Training/mask/img0001/63.npy"

with zipfile.ZipFile(zip_path, "r") as z:
    with Image.open(BytesIO(z.read(image_name))) as im:
        image = np.array(im.convert("RGB"))

    mask = np.load(
        BytesIO(z.read(mask_name)),
        allow_pickle=False
    )

# 检查三个 RGB 通道是否相同
print(
    "R == G:",
    np.array_equal(image[:, :, 0], image[:, :, 1])
)

print(
    "G == B:",
    np.array_equal(image[:, :, 1], image[:, :, 2])
)

# 创建可视化
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

axes[0].imshow(image)
axes[0].set_title("CT Image (369 x 369)")

axes[1].imshow(mask, cmap="viridis", vmin=0, vmax=6)
axes[1].set_title("Mask (512 x 512)")

# 仅用于诊断的空间对照，不保存为训练标签
mask_preview = Image.fromarray(mask.astype(np.uint8))
mask_preview = mask_preview.resize(
    (image.shape[1], image.shape[0]),
    resample=Image.Resampling.NEAREST
)

axes[2].imshow(image)
axes[2].imshow(
    np.ma.masked_equal(np.asarray(mask_preview), 0),
    cmap="autumn",
    alpha=0.45,
    interpolation="nearest"
)
axes[2].set_title("Diagnostic Overlay (assumes resize)")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

output = Path(
    "/mnt/gptdata/maoyutao/code/nnunet-practice/btcv_alignment_check.png"
)

plt.savefig(output, dpi=200)
print("Saved:", output)
