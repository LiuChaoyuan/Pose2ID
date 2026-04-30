import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch
from torch.utils.data import Dataset, DataLoader

DWPOSE_ROOT = Path("/root/DWPose/ControlNet-v1-1-nightly")
DATA_ROOT = Path("/root/autodl-fs/datasets/market1501")

os.chdir(str(DWPOSE_ROOT))
sys.path.insert(0, str(DWPOSE_ROOT))

from annotator.dwpose import DWposeDetector

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(path: Path):
    return path.is_file() and path.suffix.lower() in IMG_EXTS


class MarketDataset(Dataset):
    def __init__(self, input_dir):
        self.image_paths = sorted([p for p in input_dir.iterdir() if is_image_file(p)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
            return str(img_path), img_np
        except Exception:
            return str(img_path), np.array([])


def save_image_async(pose_img, save_path):
    try:
        if isinstance(pose_img, np.ndarray):
            pose_pil = Image.fromarray(pose_img)
        elif isinstance(pose_img, Image.Image):
            pose_pil = pose_img
        else:
            return False

        pose_pil.save(save_path)
        return True

    except Exception as e:
        print(f"\n[WARN] Save failed for {save_path}: {e}")
        return False


def process_split(detector, split_name: str, num_workers=8, save_workers=8):
    input_dir = DATA_ROOT / split_name
    output_dir = DATA_ROOT / f"{split_name}_pose"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MarketDataset(input_dir)

    if len(dataset) == 0:
        print(f"\n[SKIP] No images found in {input_dir}")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=lambda x: x[0],
    )

    print(f"\n[INFO] ---------------------------------------")
    print(f"[INFO] Processing split: {split_name}")
    print(f"[INFO] Input dir : {input_dir}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Num images: {len(dataset)}")
    print(f"[INFO] DataLoader workers: {num_workers}")
    print(f"[INFO] Save workers: {save_workers}")

    success = 0
    failed = 0

    save_executor = ThreadPoolExecutor(max_workers=save_workers)
    futures = []

    for img_path_str, img_np in tqdm(
        dataloader,
        desc=split_name,
        total=len(dataset),
        ncols=100,
    ):
        img_path = Path(img_path_str)
        save_path = output_dir / img_path.name

        if img_np.size == 0:
            failed += 1
            continue

        try:
            with torch.inference_mode():
                result = detector(img_np)

            pose_img = result[0] if isinstance(result, tuple) else result

            if pose_img is None:
                failed += 1
                continue

            futures.append(save_executor.submit(save_image_async, pose_img, save_path))
            success += 1

            if len(futures) >= 512:
                for f in futures:
                    ok = f.result()
                    if not ok:
                        failed += 1
                        success -= 1
                futures.clear()

        except Exception as e:
            failed += 1
            print(f"\n[WARN] Inference failed on {img_path.name}: {e}")

    print(f"\n[INFO] Inference finished. Waiting for disk writing to complete...")

    for f in futures:
        ok = f.result()
        if not ok:
            failed += 1
            success -= 1
    futures.clear()

    save_executor.shutdown(wait=True)

    print(f"[DONE] {split_name}: success={success}, failed={failed}\n")


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print("[INFO] CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("[INFO] GPU:", torch.cuda.get_device_name(0))

    print("[INFO] Loading DWPose Detector...")
    detector = DWposeDetector()

    process_split(detector, "bounding_box_train", num_workers=8, save_workers=8)
    process_split(detector, "bounding_box_test", num_workers=8, save_workers=8)
    process_split(detector, "query", num_workers=8, save_workers=8)


if __name__ == "__main__":
    main()