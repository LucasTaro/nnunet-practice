
from pathlib import Path
from collections import Counter
import zipfile

zip_path = Path(
    "/mnt/gptdata/maoyutao/data/"
    "TotalsegmentatorMRI_dataset_v200.zip"
)

with zipfile.ZipFile(zip_path, "r") as z:
    files = [
        f for f in z.infolist()
        if not f.is_dir()
        and not f.filename.startswith("__MACOSX/")
        and ".DS_Store" not in f.filename
    ]

    print("有效文件数量:", len(files))

    total_size = sum(f.file_size for f in files)
    print(f"预计解压大小: {total_size / 1024**3:.2f} GiB")

    extensions = Counter()

    for f in files:
        name = f.filename.lower()

        if name.endswith(".nii.gz"):
            extensions[".nii.gz"] += 1
        else:
            extensions[Path(name).suffix or "(无后缀)"] += 1

    print("\n文件格式统计:")
    for ext, count in extensions.most_common():
        print(f"{ext}: {count}")

    print("\n前 40 个文件:")
    for f in files[:40]:
        print(f.filename)

    print("\n体积最大的 10 个文件:")
    largest = sorted(
        files,
        key=lambda f: f.file_size,
        reverse=True
    )[:10]

    for f in largest:
        print(
            f"{f.file_size / 1024**2:.1f} MB",
            f.filename
        )
