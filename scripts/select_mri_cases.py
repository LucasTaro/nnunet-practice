
import csv
import gzip
import io
import zipfile
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

ZIP_PATH = Path(
    "/mnt/gptdata/maoyutao/data/"
    "TotalsegmentatorMRI_dataset_v200.zip"
)

TARGET_ORGANS = [
    "liver",
    "spleen",
    "kidney_left",
]

# 最多抽查 30 个训练病例
MAX_CASES = 30


def foreground_count(z, name):
    """从 ZIP 中读取一个 NIfTI Mask，统计前景体素数。"""
    raw = z.read(name)
    nii_bytes = gzip.decompress(raw)

    img = nib.Nifti1Image.from_bytes(nii_bytes)
    mask = np.asanyarray(img.dataobj)

    return int(np.count_nonzero(mask))


with zipfile.ZipFile(ZIP_PATH) as z:

    names = set(z.namelist())

    # 1. 正确解析 CSV
    with z.open("meta.csv") as f:
        text = io.TextIOWrapper(
            f, encoding="utf-8-sig"
        )
        reader = csv.DictReader(
            text, delimiter=";"
        )
        records = list(reader)

    print("CSV 字段:", list(records[0].keys()))
    print("病例总数:", len(records))

    split_counts = Counter(
        row["split"].strip().lower()
        for row in records
    )

    print("官方划分:", dict(split_counts))

    # 2. 查看 s0299 中有哪些非空器官
    print("\n===== s0299 的非空器官 =====")

    prefix = "s0299/segmentations/"

    organ_files = sorted(
        name for name in names
        if name.startswith(prefix)
        and name.endswith(".nii.gz")
    )

    for name in organ_files:
        count = foreground_count(z, name)

        if count > 0:
            organ = Path(name).name.removesuffix(".nii.gz")
            print(f"{organ}: {count} voxels")

    # 3. 仅抽查官方训练病例
    train_ids = sorted(
        row["image_id"]
        for row in records
        if row["split"].strip().lower() == "train"
        and row["image_id"] != "s0299"
    )

    print("\n===== 抽查训练病例 =====")

    found = []

    for case_id in train_ids[:MAX_CASES]:

        counts = {}

        for organ in TARGET_ORGANS:
            name = (
                f"{case_id}/segmentations/"
                f"{organ}.nii.gz"
            )

            if name not in names:
                counts[organ] = None
                continue

            counts[organ] = foreground_count(z, name)

        print(case_id, counts)

        # 三个器官都具有非空标注
        if all(
            counts[o] is not None and counts[o] > 0
            for o in TARGET_ORGANS
        ):
            found.append(case_id)

    print("\n三个目标器官均非空的病例:")
    print(found)
