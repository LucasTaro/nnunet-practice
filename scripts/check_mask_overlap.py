
from pathlib import Path
from io import BytesIO
import zipfile
import numpy as np
import nibabel as nib
import gzip

ZIP_PATH = Path(
    "/mnt/gptdata/maoyutao/data/"
    "TotalsegmentatorMRI_dataset_v200.zip"
)

CASE_ID = "s0009"

ORGANS = ["liver", "spleen", "kidney_left"]


def load_mask(z, organ):
    filename = (
        f"{CASE_ID}/segmentations/{organ}.nii.gz"
    )

    raw = gzip.decompress(z.read(filename))
    img = nib.Nifti1Image.from_bytes(raw)

    return np.asarray(img.dataobj) > 0


with zipfile.ZipFile(ZIP_PATH) as z:
    masks = {
        organ: load_mask(z, organ)
        for organ in ORGANS
    }

# 两两计算重叠
for i, organ_a in enumerate(ORGANS):
    for organ_b in ORGANS[i + 1:]:

        mask_a = masks[organ_a]
        mask_b = masks[organ_b]

        intersection = mask_a & mask_b

        overlap_count = int(intersection.sum())

        size_a = int(mask_a.sum())
        size_b = int(mask_b.sum())

        print(f"\n{organ_a} vs {organ_b}")
        print("Overlap voxels:", overlap_count)

        print(
            "Percentage of smaller organ:",
            f"{overlap_count / max(min(size_a, size_b), 1) * 100:.4f}%"
        )

        if overlap_count > 0:
            coordinates = np.argwhere(intersection)

            print(
                "Bounding box min:",
                coordinates.min(axis=0)
            )
            print(
                "Bounding box max:",
                coordinates.max(axis=0)
            )

            z_slices = np.unique(coordinates[:, 2])
            print("Affected slices:", z_slices)
