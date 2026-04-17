import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

# =========================
# 路径配置
# =========================
DWPOSE_ROOT = Path("/root/DWPose/ControlNet-v1-1-nightly")
DATA_ROOT = Path("/root/datasets/Market-1501-v15.09.15")

# 很重要：切到 DWPose 根目录，避免相对路径问题
os.chdir(str(DWPOSE_ROOT))
sys.path.insert(0, str(DWPOSE_ROOT))

from annotator.dwpose import DWposeDetector

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(path: Path):
    return path.is_file() and path.suffix.lower() in IMG_EXTS


def process_split(detector, split_name: str):
    input_dir = DATA_ROOT / split_name
    output_dir = DATA_ROOT / f"{split_name}_pose"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in input_dir.iterdir() if is_image_file(p)])

    print(f"\n[INFO] Processing split: {split_name}")
    print(f"[INFO] Input dir : {input_dir}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Num images: {len(image_paths)}")

    success = 0
    failed = 0

    for img_path in tqdm(image_paths, desc=split_name):
        save_path = output_dir / img_path.name

        try:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)

            result = detector(img_np)
            pose_img = result[0] if isinstance(result, tuple) else result

            if pose_img is None:
                failed += 1
                continue

            if isinstance(pose_img, np.ndarray):
                pose_pil = Image.fromarray(pose_img)
            elif isinstance(pose_img, Image.Image):
                pose_pil = pose_img
            else:
                failed += 1
                continue

            pose_pil.save(save_path)
            success += 1

        except Exception as e:
            failed += 1
            print(f"[WARN] Failed on {img_path.name}: {e}")

    print(f"[DONE] {split_name}: success={success}, failed={failed}")


def main():
    detector = DWposeDetector()

    # 先只处理训练集
    process_split(detector, "bounding_box_train_tmp")

    # 如果以后需要，也可以打开这两行
    # process_split(detector, "bounding_box_test")
    # process_split(detector, "query")


if __name__ == "__main__":
    main()