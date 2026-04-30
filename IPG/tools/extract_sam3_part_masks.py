from __future__ import annotations

"""
Extract upper/lower body semantic masks with SAM3 text prompts.

The script writes masks in a layout consumed by the updated IPG dataset and
Market_gen.py:

    output_root/<split>/upper/<relative-image-path>
    output_root/<split>/lower/<relative-image-path>

Example:
    python tools/extract_sam3_part_masks.py \
        --data_root /root/autodl-fs/datasets/market1501 \
        --splits bounding_box_train bounding_box_test query \
        --standard_pose_dir ./standard_poses
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

try:
    import torch
except ImportError:  # pragma: no cover - handled at runtime.
    torch = None

try:
    import cv2
except ImportError:  # pragma: no cover - PIL fallback is used instead.
    cv2 = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_UPPER_PROMPTS = [
    "head",
    "hair",
    "face",
    "neck",
    "upper body clothing",
    "shirt",
    "coat",
    "dress",
    "arm",
    "hand",
]

DEFAULT_LOWER_PROMPTS = [
    "pants",
    "skirt",
    "leg",
    "shoe",
    "foot",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract SAM3 upper/lower semantic masks for Pose2ID IPG."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/autodl-fs/datasets/market1501",
        help="Dataset root containing Market1501 split folders.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["bounding_box_train", "bounding_box_test", "query"],
        help="Split folders under data_root to process.",
    )
    parser.add_argument(
        "--input_dirs",
        nargs="*",
        default=None,
        help="Optional explicit image directories. If set, --splits is ignored.",
    )
    parser.add_argument(
        "--standard_pose_dir",
        type=str,
        default=None,
        help="Optional standard pose directory to process as split 'standard_poses'.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/root/autodl-fs/datasets/market1501/sam3_part_masks",
    )
    parser.add_argument("--model_name", type=str, default="facebook/sam3")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--min_area_ratio", type=float, default=0.0005)
    parser.add_argument("--dilate", type=int, default=5)
    parser.add_argument("--blur", type=int, default=9)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--upper_prompts", nargs="+", default=DEFAULT_UPPER_PROMPTS)
    parser.add_argument("--lower_prompts", nargs="+", default=DEFAULT_LOWER_PROMPTS)
    parser.add_argument(
        "--fallback",
        choices=["half", "blank"],
        default="half",
        help="Mask to write when SAM3 finds no prompt instances.",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_jobs(args) -> List[Tuple[str, Path]]:
    if args.input_dirs:
        return [(Path(path).name, Path(path)) for path in args.input_dirs]

    data_root = Path(args.data_root)
    jobs = [(split, data_root / split) for split in args.splits]
    if args.standard_pose_dir:
        jobs.append(("standard_poses", Path(args.standard_pose_dir)))
    return jobs


def load_sam3(model_name: str, device_arg: str):
    if torch is None:
        raise ImportError(
            "SAM3 mask extraction requires PyTorch. Install torch in the mask "
            "extraction environment before running this tool."
        )
    try:
        from transformers import Sam3Model, Sam3Processor
    except ImportError as exc:
        raise ImportError(
            "SAM3 requires a Transformers version that provides Sam3Model and "
            "Sam3Processor. Install a recent Transformers release or source build "
            "before running this tool."
        ) from exc

    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg

    model = Sam3Model.from_pretrained(model_name).to(device)
    processor = Sam3Processor.from_pretrained(model_name)
    model.eval()
    return model, processor, torch.device(device)


def to_numpy_mask(mask) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[0]
    return mask.astype(np.float32)


def filter_and_union_masks(
    results: Dict,
    height: int,
    width: int,
    *,
    min_area_ratio: float,
) -> np.ndarray:
    masks = results.get("masks", [])
    scores = results.get("scores", None)
    if isinstance(masks, torch.Tensor):
        masks_iter = list(masks)
    else:
        masks_iter = list(masks)

    if scores is None:
        scores_iter = [1.0] * len(masks_iter)
    elif isinstance(scores, torch.Tensor):
        scores_iter = scores.detach().cpu().float().tolist()
    else:
        scores_iter = list(scores)

    min_area = float(height * width) * min_area_ratio
    union = np.zeros((height, width), dtype=np.float32)
    for mask, _score in zip(masks_iter, scores_iter):
        mask_np = to_numpy_mask(mask)
        if mask_np.shape != (height, width):
            mask_np = np.array(
                Image.fromarray((mask_np > 0.5).astype(np.uint8) * 255).resize(
                    (width, height), Image.Resampling.NEAREST
                ),
                dtype=np.float32,
            ) / 255.0
        mask_np = (mask_np > 0.5).astype(np.float32)
        if mask_np.sum() < min_area:
            continue
        union = np.maximum(union, mask_np)
    return union


def fallback_masks(height: int, width: int, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "blank":
        blank = np.zeros((height, width), dtype=np.float32)
        return blank, blank.copy()

    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    upper = 1.0 / (1.0 + np.exp((y - 0.5) / 0.08))
    lower = 1.0 / (1.0 + np.exp((0.5 - y) / 0.08))
    return (
        np.repeat(upper, width, axis=1).astype(np.float32),
        np.repeat(lower, width, axis=1).astype(np.float32),
    )


def postprocess_mask(mask: np.ndarray, dilate: int, blur: int) -> np.ndarray:
    mask = np.clip(mask, 0, 1).astype(np.float32)
    if cv2 is not None:
        mask_u8 = (mask * 255).astype(np.uint8)
        if dilate > 1:
            kernel = np.ones((dilate, dilate), np.uint8)
            mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
        if blur > 1:
            if blur % 2 == 0:
                blur += 1
            mask_u8 = cv2.GaussianBlur(mask_u8, (blur, blur), 0)
        return mask_u8.astype(np.float32) / 255.0

    image = Image.fromarray((mask * 255).astype(np.uint8))
    if dilate > 1:
        image = image.filter(ImageFilter.MaxFilter(dilate if dilate % 2 == 1 else dilate + 1))
    if blur > 1:
        image = image.filter(ImageFilter.GaussianBlur(radius=max(1, blur // 3)))
    return np.asarray(image, dtype=np.float32) / 255.0


def save_mask(mask: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8)).save(path)


def run_prompt(
    model,
    processor,
    device: torch.device,
    image: Image.Image,
    prompt: str,
    *,
    threshold: float,
    mask_threshold: float,
    min_area_ratio: float,
) -> np.ndarray:
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    width, height = image.size
    return filter_and_union_masks(
        results, height, width, min_area_ratio=min_area_ratio
    )


def run_prompt_group(
    model,
    processor,
    device: torch.device,
    image: Image.Image,
    prompts: Sequence[str],
    args,
) -> np.ndarray:
    width, height = image.size
    union = np.zeros((height, width), dtype=np.float32)
    for prompt in prompts:
        try:
            mask = run_prompt(
                model,
                processor,
                device,
                image,
                prompt,
                threshold=args.threshold,
                mask_threshold=args.mask_threshold,
                min_area_ratio=args.min_area_ratio,
            )
            union = np.maximum(union, mask)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
    return union


def process_image(model, processor, device, image_path: Path, args) -> Tuple[np.ndarray, np.ndarray]:
    image = Image.open(image_path).convert("RGB")
    upper = run_prompt_group(model, processor, device, image, args.upper_prompts, args)
    lower = run_prompt_group(model, processor, device, image, args.lower_prompts, args)

    if upper.max() <= 0 and lower.max() <= 0:
        upper, lower = fallback_masks(image.height, image.width, args.fallback)
    elif upper.max() <= 0:
        fallback_upper, _ = fallback_masks(image.height, image.width, args.fallback)
        upper = fallback_upper
    elif lower.max() <= 0:
        _, fallback_lower = fallback_masks(image.height, image.width, args.fallback)
        lower = fallback_lower

    upper = postprocess_mask(upper, args.dilate, args.blur)
    lower = postprocess_mask(lower, args.dilate, args.blur)
    return upper, lower


def process_split(model, processor, device, split_name: str, input_dir: Path, args) -> Dict:
    if not input_dir.exists():
        print(f"[SKIP] Input dir does not exist: {input_dir}")
        return {"split": split_name, "processed": 0, "skipped": 0, "failed": 0}

    image_paths = sorted(path for path in input_dir.rglob("*") if is_image_file(path))
    output_root = Path(args.output_root) / split_name
    processed = 0
    skipped = 0
    failed = 0

    print(f"[INFO] Processing {split_name}: {input_dir} -> {output_root}")
    for image_path in tqdm(image_paths, desc=split_name, ncols=100):
        rel_path = image_path.relative_to(input_dir)
        upper_path = output_root / "upper" / rel_path
        lower_path = output_root / "lower" / rel_path
        if not args.overwrite and upper_path.exists() and lower_path.exists():
            skipped += 1
            continue

        try:
            upper, lower = process_image(model, processor, device, image_path, args)
            save_mask(upper, upper_path)
            save_mask(lower, lower_path)
            processed += 1
        except Exception as exc:
            failed += 1
            print(f"\n[WARN] Failed on {image_path}: {exc}")

    return {
        "split": split_name,
        "input_dir": str(input_dir),
        "output_dir": str(output_root),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
    }


def write_manifest(args, summaries: Sequence[Dict]):
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_name": args.model_name,
        "upper_prompts": args.upper_prompts,
        "lower_prompts": args.lower_prompts,
        "threshold": args.threshold,
        "mask_threshold": args.mask_threshold,
        "min_area_ratio": args.min_area_ratio,
        "splits": list(summaries),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    args = parse_args()
    model, processor, device = load_sam3(args.model_name, args.device)
    summaries = []
    for split_name, input_dir in collect_jobs(args):
        summaries.append(process_split(model, processor, device, split_name, input_dir, args))
    write_manifest(args, summaries)
    print("[DONE] SAM3 part mask extraction complete.")


if __name__ == "__main__":
    main()
