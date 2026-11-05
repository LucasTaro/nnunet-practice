
import csv
import gzip
import io
import json
import shutil
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

ZIP_PATH = Path(
    "/mnt/gptdata/maoyutao/data/"
    "TotalsegmentatorMRI_dataset_v200.zip"
)

OUTPUT_ROOT = Path(
    "/mnt/gptdata/maoyutao/data/"
    "nnUNet_raw/Dataset102_TotalSegMRI3Organs"
)

# 使用已经检查过的 23 个训练病例
CASE_IDS = [
    "s0001", "s0005", "s0006", "s0007", "s0009",
    "s0010", "s0011", "s0012", "s0014", "s0017",
    "s0018", "s0019", "s0020", "s0022", "s0025",
    "s0028", "s0029", "s0030", "s0032", "s0033",
    "s0034", "s0035", "s0036"
]

ORGANS = {
    "liver": 1,
    "spleen": 2,
    "kidney_left": 3,
}


def load_nii_from_zip(z, filename):
    """读取 ZIP 内部的 .nii.gz 文件。"""
    compressed = z.read(filename)
    nii_bytes = gzip.decompress(compressed)
    return nib.Nifti1Image.from_bytes(nii_bytes)



def convert_case(z, case_id, image_dir, label_dir):
    print(f"\nProcessing {case_id}")

    image_name = f"{case_id}/mri.nii.gz"

    # 先在内存中完成全部检查，成功后才写文件
    image = load_nii_from_zip(z, image_name)

    if len(image.shape) != 3:
        raise ValueError(f"{case_id}: expected 3D MRI")

    masks = {}

    for organ, class_id in ORGANS.items():
        mask_name = (
            f"{case_id}/segmentations/{organ}.nii.gz"
        )

        mask_img = load_nii_from_zip(z, mask_name)
        mask = np.asarray(mask_img.dataobj)

        if mask.shape != image.shape:
            raise ValueError(
                f"{case_id}: {organ} shape mismatch"
            )

        if not np.allclose(
            mask_img.affine, image.affine, atol=1e-4
        ):
            raise ValueError(
                f"{case_id}: {organ} affine mismatch"
            )

        if not np.all(np.isin(mask, [0, 1])):
            raise ValueError(
                f"{case_id}: {organ} is not binary"
            )

        masks[organ] = mask > 0

    # 统计每个体素属于几个器官
    overlap_count = np.zeros(image.shape, dtype=np.uint8)

    for mask in masks.values():
        overlap_count += mask.astype(np.uint8)

    overlap = overlap_count > 1

    print(
        "  Overlapping voxels:",
        int(np.count_nonzero(overlap))
    )

    # 合并标签，冲突体素暂时设为背景
    label = np.zeros(image.shape, dtype=np.uint8)

    for organ, class_id in ORGANS.items():
        foreground = masks[organ] & ~overlap
        label[foreground] = class_id

        print(
            f"  {organ}: "
            f"{np.count_nonzero(foreground)} voxels"
        )

    # 保留原始 NIfTI 空间信息
    header = image.header.copy()
    header.set_data_dtype(np.uint8)

    label_img = nib.Nifti1Image(
        label,
        affine=image.affine,
        header=header
    )

    # 全部检查完成后才写出文件
    output_image = image_dir / f"{case_id}_0000.nii.gz"
    output_label = label_dir / f"{case_id}.nii.gz"

    with z.open(image_name) as src:
        with open(output_image, "wb") as dst:
            shutil.copyfileobj(src, dst)

    nib.save(label_img, str(output_label))

    print("  Saved:", output_label.name)

    return int(np.count_nonzero(overlap))



def main():
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {OUTPUT_ROOT}"
        )

    image_dir = OUTPUT_ROOT / "imagesTr"
    label_dir = OUTPUT_ROOT / "labelsTr"

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        available = set(z.namelist())

        # 检查所有病例确实来自官方训练集
        with z.open("meta.csv") as f:
            reader = csv.DictReader(
                io.TextIOWrapper(f, encoding="utf-8-sig"),
                delimiter=";"
            )

            official_train = {
                row["image_id"]
                for row in reader
                if row["split"].strip().lower() == "train"
            }

        if not set(CASE_IDS).issubset(official_train):
            raise ValueError(
                "Some cases are not in the official training split"
            )

        # 预先检查所有文件是否存在
        for case_id in CASE_IDS:
            required = [f"{case_id}/mri.nii.gz"]
            required += [
                f"{case_id}/segmentations/{organ}.nii.gz"
                for organ in ORGANS
            ]

            missing = [p for p in required if p not in available]
            if missing:
                raise FileNotFoundError(missing)

        for case_id in CASE_IDS:
            convert_case(z, case_id, image_dir, label_dir)

    # 6. 创建 dataset.json
    dataset = {
        "channel_names": {
            "0": "MRI"
        },
        "labels": {
            "background": 0,
            "liver": 1,
            "spleen": 2,
            "kidney_left": 3
        },
        "numTraining": len(CASE_IDS),
        "file_ending": ".nii.gz"
    }

    with open(
        OUTPUT_ROOT / "dataset.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(dataset, f, indent=2)

    print("\nDataset conversion completed!")
    print("Number of cases:", len(CASE_IDS))
    print("Output:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
