
import zipfile
from pathlib import Path
from collections import Counter

zip_path = Path("/mnt/gptdata/maoyutao/data/btcv.zip")

with zipfile.ZipFile(zip_path, "r") as z:
    files = [
        f for f in z.infolist()
        if not f.is_dir()
        and "__MACOSX" not in f.filename
        and ".DS_Store" not in f.filename
    ]

    print("有效文件数量:", len(files))

    # 统计文件后缀
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

    # 每个目录只显示少量文件
    folders = [
        "btcv/Training/image/img0001/",
        "btcv/Training/mask/img0001/",
        "btcv/Test/image/img0003/",
        "btcv/Test/mask/img0003/",
    ]

    for folder in folders:
        matches = [
            f for f in files
            if f.filename.startswith(folder)
        ]

        print(f"\n{folder}")
        print("文件数量:", len(matches))

        for f in matches[:5]:
            print(
                f.filename,
                f"({f.file_size / 1024:.1f} KB)"
            )
