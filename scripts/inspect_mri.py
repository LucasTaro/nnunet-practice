
from pathlib import Path
import nibabel as nib
import numpy as np

root = Path(
    "/mnt/gptdata/maoyutao/data/"
    "totalseg_exploration/s0299"
)

files = [
    "mri.nii.gz",
    "liver.nii.gz",
    "spleen.nii.gz",
    "kidney_left.nii.gz",
]

reference = nib.load(str(root / "mri.nii.gz"))

for filename in files:
    img = nib.load(str(root / filename))

    print(f"\n===== {filename} =====")

    print("Shape:", img.shape)
    print("Spacing:", img.header.get_zooms()[:3])
    print("Data type:", img.get_data_dtype())
    print("Affine:")
    print(img.affine)

    if filename != "mri.nii.gz":
        mask = np.asarray(img.dataobj)

        print("Unique labels:", np.unique(mask))
        print("Foreground voxels:", np.count_nonzero(mask))

        print(
            "Same shape:",
            img.shape == reference.shape
        )

        print(
            "Same affine:",
            np.allclose(
                img.affine,
                reference.affine,
                atol=1e-4
            )
        )
