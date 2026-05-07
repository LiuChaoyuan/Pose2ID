from __future__ import annotations

"""
Extract upper/lower body semantic masks with SAM3 text prompts.
Optimized for vGPU-48GB + 20 vCPU.
Fix: Sam3Processor does not support padding= argument;
     inputs are encoded per-image and stacked manually.
"""

import argparse
import json
import os
import queue
import threading
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None

try:
    import cv2
except ImportError:
    cv2 = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_UPPER_PROMPTS = [
    "head", "hair", "face", "neck",
    "upper body clothing", "shirt", "coat", "dress", "arm", "hand",
]
DEFAULT_LOWER_PROMPTS = [
    "pants", "skirt", "leg", "shoe", "foot",
]


# ──────────────────────────────────────────────
# 参数
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract SAM3 upper/lower semantic masks (optimized)."
    )
    parser.add_argument("--data_root", type=str,
                        default="/root/autodl-tmp/datasets/market1501_less")
    parser.add_argument("--splits", nargs="+",
                        default=["bounding_box_test_less", "query_less"])
    parser.add_argument("--input_dirs", nargs="*", default=None)
    parser.add_argument("--standard_pose_dir", type=str, default=None)
    parser.add_argument("--output_root", type=str,
                        default="/root/autodl-tmp/datasets/market1501_less/sam3_part_masks")
    parser.add_argument("--model_name", type=str,
                        default="/root/autodl-tmp/checkpoints/sam3")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--min_area_ratio", type=float, default=0.0005)
    parser.add_argument("--dilate", type=int, default=5)
    parser.add_argument("--blur", type=int, default=9)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--upper_prompts", nargs="+", default=DEFAULT_UPPER_PROMPTS)
    parser.add_argument("--lower_prompts", nargs="+", default=DEFAULT_LOWER_PROMPTS)
    parser.add_argument("--fallback", choices=["half", "blank"], default="half")

    # ── 性能参数 ──
    parser.add_argument("--batch_size", type=int, default=8,
                        help="每批图像数 (SAM3 单图编码，此处控制并发流水线深度)。")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader 读图并行进程数。")
    parser.add_argument("--save_workers", type=int, default=4,
                        help="异步保存线程数。")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="启用混合精度推理 (默认开启)。")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    return parser.parse_args()


# ──────────────────────────────────────────────
# 工具
# ──────────────────────────────────────────────

def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_jobs(args) -> List[Tuple[str, Path]]:
    if args.input_dirs:
        return [(Path(p).name, Path(p)) for p in args.input_dirs]
    data_root = Path(args.data_root)
    jobs = [(s, data_root / s) for s in args.splits]
    if args.standard_pose_dir:
        jobs.append(("standard_poses", Path(args.standard_pose_dir)))
    return jobs


# ──────────────────────────────────────────────
# 模型加载
# ──────────────────────────────────────────────

def load_sam3(model_name: str, device_arg: str):
    if torch is None:
        raise ImportError("PyTorch is required.")
    try:
        from transformers import Sam3Model, Sam3Processor
    except ImportError as exc:
        raise ImportError(
            "Need Transformers with Sam3Model/Sam3Processor."
        ) from exc

    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if device_arg == "auto" else device_arg

    print(f"[INFO] Loading SAM3 on {device} …")
    model = Sam3Model.from_pretrained(model_name).to(device)
    processor = Sam3Processor.from_pretrained(model_name)
    model.eval()

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    return model, processor, torch.device(device)


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class ImageDataset(Dataset):
    def __init__(self, image_paths: List[Path]):
        self.paths = image_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return np.asarray(img, dtype=np.uint8), str(path), True
        except Exception:  # noqa: BLE001
            return np.zeros((1, 1, 3), dtype=np.uint8), str(path), False


def _collate_images(batch):
    """不堆叠图像张量，保持列表形式（各图尺寸不同）。"""
    images, paths, valids = zip(*batch)
    return list(images), list(paths), list(valids)


# ──────────────────────────────────────────────
# 异步保存
# ──────────────────────────────────────────────

_SENTINEL = object()


def _save_worker(q: queue.Queue):
    while True:
        item = q.get()
        if item is _SENTINEL:
            q.task_done()
            break
        mask, path = item
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(
                (np.clip(mask, 0, 1) * 255).astype(np.uint8)
            ).save(path)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Save failed {path}: {exc}")
        finally:
            q.task_done()


class AsyncSaver:
    def __init__(self, num_workers: int = 4):
        self._q: queue.Queue = queue.Queue(maxsize=num_workers * 32)
        self._threads = []
        for _ in range(num_workers):
            t = threading.Thread(target=_save_worker, args=(self._q,), daemon=True)
            t.start()
            self._threads.append(t)

    def save(self, mask: np.ndarray, path: Path):
        self._q.put((mask, path))

    def join(self):
        for _ in self._threads:
            self._q.put(_SENTINEL)
        self._q.join()
        for t in self._threads:
            t.join()


# ──────────────────────────────────────────────
# mask 工具
# ──────────────────────────────────────────────

def to_numpy_mask(mask) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[0]
    return mask


def filter_and_union_masks(
    results: Dict,
    height: int,
    width: int,
    min_area_ratio: float,
) -> np.ndarray:
    masks = list(results.get("masks", []))
    scores = results.get("scores", None)

    if scores is None:
        scores_iter = [1.0] * len(masks)
    elif isinstance(scores, torch.Tensor):
        scores_iter = scores.detach().cpu().float().tolist()
    else:
        scores_iter = list(scores)

    min_area = float(height * width) * min_area_ratio
    union = np.zeros((height, width), dtype=np.float32)
    for mask, _ in zip(masks, scores_iter):
        mask_np = to_numpy_mask(mask)
        if mask_np.shape != (height, width):
            mask_np = np.array(
                Image.fromarray(
                    (mask_np > 0.5).astype(np.uint8) * 255
                ).resize((width, height), Image.Resampling.NEAREST),
                dtype=np.float32,
            ) / 255.0
        mask_np = (mask_np > 0.5).astype(np.float32)
        if mask_np.sum() < min_area:
            continue
        union = np.maximum(union, mask_np)
    return union


def fallback_masks(height: int, width: int, mode: str):
    if mode == "blank":
        blank = np.zeros((height, width), dtype=np.float32)
        return blank, blank.copy()
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    upper = 1.0 / (1.0 + np.exp((y - 0.5) / 0.08))
    lower = 1.0 / (1.0 + np.exp((0.5 - y) / 0.08))
    return np.repeat(upper, width, axis=1), np.repeat(lower, width, axis=1)


def postprocess_mask(mask: np.ndarray, dilate: int, blur: int) -> np.ndarray:
    mask = np.clip(mask, 0, 1).astype(np.float32)
    if cv2 is not None:
        mask_u8 = (mask * 255).astype(np.uint8)
        if dilate > 1:
            kernel = np.ones((dilate, dilate), np.uint8)
            mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
        if blur > 1:
            blur = blur if blur % 2 == 1 else blur + 1
            mask_u8 = cv2.GaussianBlur(mask_u8, (blur, blur), 0)
        return mask_u8.astype(np.float32) / 255.0
    img = Image.fromarray((mask * 255).astype(np.uint8))
    if dilate > 1:
        img = img.filter(
            ImageFilter.MaxFilter(dilate if dilate % 2 == 1 else dilate + 1)
        )
    if blur > 1:
        img = img.filter(ImageFilter.GaussianBlur(radius=max(1, blur // 3)))
    return np.asarray(img, dtype=np.float32) / 255.0


# ──────────────────────────────────────────────
# 核心推理：单图 × 单 prompt
# ──────────────────────────────────────────────

def _infer_one(
    model,
    processor,
    device: torch.device,
    pil_image: Image.Image,
    prompt: str,
    args,
) -> np.ndarray:
    """
    对单张图像运行单条 prompt，返回 float32 mask (H×W)。
    Sam3Processor 只接受单图单文本，不支持 padding 批量化。
    """
    inputs = processor(
        images=pil_image,
        text=prompt,
        return_tensors="pt",
    ).to(device)

    amp_enabled = args.amp and device.type == "cuda"
    ctx = torch.autocast(device_type="cuda", dtype=torch.float16) \
        if amp_enabled else torch.no_grad()

    with torch.no_grad(), ctx:
        outputs = model(**inputs)

    target_sizes = inputs.get("original_sizes").tolist()
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=args.threshold,
        mask_threshold=args.mask_threshold,
        target_sizes=target_sizes,
    )[0]

    w, h = pil_image.size  # PIL: (width, height)
    return filter_and_union_masks(results, h, w, min_area_ratio=args.min_area_ratio)


# ──────────────────────────────────────────────
# 批量推理：N 张图 × M 个 prompt（串行 prompt，并行图用线程）
# ──────────────────────────────────────────────

def _run_prompt_group_for_image(
    model,
    processor,
    device: torch.device,
    pil_image: Image.Image,
    prompts: Sequence[str],
    args,
) -> np.ndarray:
    """单张图跑完所有 prompts，返回并集 mask。"""
    w, h = pil_image.size
    union = np.zeros((h, w), dtype=np.float32)
    for prompt in prompts:
        mask = _infer_one(model, processor, device, pil_image, prompt, args)
        union = np.maximum(union, mask)
    return union


def run_batch_prompts(
    model,
    processor,
    device: torch.device,
    pil_images: List[Image.Image],
    prompts: Sequence[str],
    args,
) -> List[np.ndarray]:
    """
    对 pil_images 里每张图，顺序跑所有 prompts。
    GPU 推理本身无法跨图并行（SAM3 不支持 padding batch），
    因此保持顺序推理；通过 DataLoader 预取 + 异步存储来掩盖等待。
    """
    results = []
    for img in pil_images:
        mask = _run_prompt_group_for_image(
            model, processor, device, img, prompts, args
        )
        results.append(mask)
    return results


# ──────────────────────────────────────────────
# Split 处理
# ──────────────────────────────────────────────

def process_split(
    model,
    processor,
    device: torch.device,
    split_name: str,
    input_dir: Path,
    args,
) -> Dict:
    if not input_dir.exists():
        print(f"[SKIP] {input_dir} does not exist.")
        return {"split": split_name, "processed": 0, "skipped": 0, "failed": 0}

    all_paths = sorted(p for p in input_dir.rglob("*") if is_image_file(p))
    output_root = Path(args.output_root) / split_name

    todo_paths, skip_count = [], 0
    for p in all_paths:
        rel = p.relative_to(input_dir)
        upper_p = output_root / "upper" / rel
        lower_p = output_root / "lower" / rel
        if not args.overwrite and upper_p.exists() and lower_p.exists():
            skip_count += 1
        else:
            todo_paths.append(p)

    print(f"[INFO] {split_name}: {len(todo_paths)} to process, "
          f"{skip_count} skipped → {output_root}")

    if not todo_paths:
        return {"split": split_name, "processed": 0,
                "skipped": skip_count, "failed": 0}

    dataset = ImageDataset(todo_paths)

    # prefetch_factor 仅在 num_workers > 0 时有效
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=_collate_images,
        pin_memory=False,          # numpy → PIL 不走 pin_memory
        persistent_workers=(args.num_workers > 0),
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4

    loader = DataLoader(dataset, **loader_kwargs)

    saver = AsyncSaver(num_workers=args.save_workers)
    processed, failed = 0, 0

    with tqdm(total=len(todo_paths), desc=split_name, ncols=110,
              dynamic_ncols=False) as pbar:
        for img_arrays, path_strs, valids in loader:

            # 过滤读图失败项
            batch_imgs: List[Image.Image] = []
            batch_paths: List[Path] = []
            for arr, ps, ok in zip(img_arrays, path_strs, valids):
                if ok:
                    batch_imgs.append(Image.fromarray(arr))
                    batch_paths.append(Path(ps))
                else:
                    failed += 1
                    pbar.update(1)

            if not batch_imgs:
                continue

            try:
                upper_masks = run_batch_prompts(
                    model, processor, device,
                    batch_imgs, args.upper_prompts, args,
                )
                lower_masks = run_batch_prompts(
                    model, processor, device,
                    batch_imgs, args.lower_prompts, args,
                )
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "out of memory" in msg and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"\n[OOM] Reduce --batch_size (current={args.batch_size}).")
                for _ in batch_imgs:
                    failed += 1
                    pbar.update(1)
                continue

            # 后处理 + 异步保存
            for img, up, lo, img_path in zip(
                batch_imgs, upper_masks, lower_masks, batch_paths
            ):
                h, w = img.height, img.width

                if up.max() <= 0 and lo.max() <= 0:
                    up, lo = fallback_masks(h, w, args.fallback)
                elif up.max() <= 0:
                    up, _ = fallback_masks(h, w, args.fallback)
                elif lo.max() <= 0:
                    _, lo = fallback_masks(h, w, args.fallback)

                up = postprocess_mask(up, args.dilate, args.blur)
                lo = postprocess_mask(lo, args.dilate, args.blur)

                rel = img_path.relative_to(input_dir)
                saver.save(up, output_root / "upper" / rel)
                saver.save(lo, output_root / "lower" / rel)
                processed += 1
                pbar.update(1)

            if device.type == "cuda":
                torch.cuda.empty_cache()

    saver.join()
    return {
        "split": split_name,
        "input_dir": str(input_dir),
        "output_dir": str(output_root),
        "processed": processed,
        "skipped": skip_count,
        "failed": failed,
    }


# ──────────────────────────────────────────────
# Manifest
# ──────────────────────────────────────────────

def write_manifest(args, summaries):
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_name": args.model_name,
        "upper_prompts": args.upper_prompts,
        "lower_prompts": args.lower_prompts,
        "threshold": args.threshold,
        "mask_threshold": args.mask_threshold,
        "min_area_ratio": args.min_area_ratio,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "amp": args.amp,
        "splits": list(summaries),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    # OMP 警告抑制：限制 OpenMP 线程数避免与 DataLoader 争抢 CPU
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    torch.set_num_threads(4)

    model, processor, device = load_sam3(args.model_name, args.device)

    summaries = []
    for split_name, input_dir in collect_jobs(args):
        summaries.append(
            process_split(model, processor, device, split_name, input_dir, args)
        )

    write_manifest(args, summaries)
    print("\n[DONE] SAM3 part mask extraction complete.")
    for s in summaries:
        print(f"  {s['split']}: processed={s['processed']} "
              f"skipped={s['skipped']} failed={s['failed']}")


if __name__ == "__main__":
    main()