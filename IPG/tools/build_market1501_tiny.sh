#!/usr/bin/env bash
set -euo pipefail

# Build a smaller Market1501-style dataset from market1501_less.
#
# Expected source layout:
#   market1501_less/
#     bounding_box_train_less
#     bounding_box_train_less_pose
#     bounding_box_test_less
#     bounding_box_test_less_pose
#     query_less
#     query_pose_less
#     clothing_colors_less_nl.json
#     sam3_part_masks/{bounding_box_train_less,bounding_box_test_less,query_less}/{upper,lower}
#
# Generated target layout:
#   market1501_tiny/
#     bounding_box_train_tiny
#     bounding_box_train_tiny_pose
#     bounding_box_test_tiny
#     bounding_box_test_tiny_pose
#     query_tiny
#     query_pose_tiny
#     clothing_colors_tiny_nl.json
#     sam3_part_masks/{bounding_box_train_tiny,bounding_box_test_tiny,query_tiny}/{upper,lower}

SRC_ROOT="${SRC_ROOT:-/root/autodl-tmp/datasets/market1501_less}"
DST_ROOT="${DST_ROOT:-/root/autodl-tmp/datasets/market1501_tiny}"

QUERY_TARGET="${QUERY_TARGET:-50}"
GALLERY_TARGET="${GALLERY_TARGET:-100}"
TRAIN_TARGET="${TRAIN_TARGET:-300}"
SEED="${SEED:-12580}"

OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

python - "$SRC_ROOT" "$DST_ROOT" "$QUERY_TARGET" "$GALLERY_TARGET" "$TRAIN_TARGET" "$SEED" "$OVERWRITE" "$DRY_RUN" <<'PY'
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def pid_of(path: Path) -> str:
    return path.name.split("_", 1)[0]


def list_images(root: Path):
    if not root.exists():
        raise FileNotFoundError(f"Missing source directory: {root}")
    return sorted(path for path in root.iterdir() if is_image(path))


def group_by_pid(paths):
    grouped = defaultdict(list)
    for path in paths:
        grouped[pid_of(path)].append(path)
    return grouped


def choose_eval_subset(query_paths, gallery_paths, query_target, gallery_target, rng):
    query_by_pid = group_by_pid(query_paths)
    gallery_by_pid = group_by_pid(gallery_paths)
    valid_pids = sorted(set(query_by_pid) & set(gallery_by_pid))
    if not valid_pids:
        raise RuntimeError("No overlapping identities between query_less and bounding_box_test_less.")

    rng.shuffle(valid_pids)
    selected_query = []
    selected_gallery = []

    # First guarantee that every selected query identity has gallery matches.
    for pid in valid_pids:
        if len(selected_query) >= query_target:
            break
        selected_query.append(rng.choice(query_by_pid[pid]))

    query_pids = {pid_of(path) for path in selected_query}
    for pid in sorted(query_pids):
        matches = gallery_by_pid[pid][:]
        rng.shuffle(matches)
        selected_gallery.extend(matches[: max(1, min(2, len(matches)))])

    remaining_gallery = [
        path
        for path in gallery_paths
        if path not in selected_gallery and pid_of(path) in query_pids
    ]
    rng.shuffle(remaining_gallery)
    selected_gallery.extend(remaining_gallery[: max(0, gallery_target - len(selected_gallery))])

    # If gallery is still short, fill with any remaining gallery images.
    if len(selected_gallery) < gallery_target:
        filler = [path for path in gallery_paths if path not in selected_gallery]
        rng.shuffle(filler)
        selected_gallery.extend(filler[: gallery_target - len(selected_gallery)])

    return sorted(selected_query[:query_target]), sorted(selected_gallery[:gallery_target])


def choose_train_subset(train_paths, train_target, rng):
    train_by_pid = group_by_pid(train_paths)
    pids = sorted(train_by_pid)
    rng.shuffle(pids)
    selected = []

    # Keep identity diversity: one image per pid first.
    for pid in pids:
        if len(selected) >= train_target:
            break
        selected.append(rng.choice(train_by_pid[pid]))

    if len(selected) < train_target:
        remaining = [path for path in train_paths if path not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: train_target - len(selected)])

    return sorted(selected[:train_target])


def copy_one(src: Path, dst: Path, dry_run: bool):
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_optional_by_name(src_dir: Path, dst_dir: Path, file_name: str, dry_run: bool):
    src = src_dir / file_name
    if src.exists():
        copy_one(src, dst_dir / file_name, dry_run)


def copy_images(paths, src_dir: Path, dst_dir: Path, dry_run: bool):
    for path in paths:
        copy_one(path, dst_dir / path.name, dry_run)


def copy_pose(paths, pose_src_dir: Path, pose_dst_dir: Path, dry_run: bool):
    for path in paths:
        copy_optional_by_name(pose_src_dir, pose_dst_dir, path.name, dry_run)


def copy_masks(paths, mask_root: Path, src_split: str, dst_split: str, dst_root: Path, dry_run: bool):
    for part in ("upper", "lower"):
        src_part_dir = mask_root / src_split / part
        dst_part_dir = dst_root / "sam3_part_masks" / dst_split / part
        for path in paths:
            copy_optional_by_name(src_part_dir, dst_part_dir, path.name, dry_run)


def filter_color_json(src_json: Path, dst_json: Path, selected_names, dry_run: bool):
    if not src_json.exists():
        return
    with src_json.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    selected = set(selected_names)
    filtered = {
        name: value
        for name, value in data.items()
        if name in selected or Path(name).name in selected or Path(name).stem in selected
    }
    if dry_run:
        return
    dst_json.parent.mkdir(parents=True, exist_ok=True)
    with dst_json.open("w", encoding="utf-8") as handle:
        json.dump(filtered, handle, ensure_ascii=False, indent=2)


def main():
    src_root = Path(sys.argv[1])
    dst_root = Path(sys.argv[2])
    query_target = int(sys.argv[3])
    gallery_target = int(sys.argv[4])
    train_target = int(sys.argv[5])
    seed = int(sys.argv[6])
    overwrite = sys.argv[7] == "1"
    dry_run = sys.argv[8] == "1"

    rng = random.Random(seed)

    src_train = src_root / "bounding_box_train_less"
    src_query = src_root / "query_less"
    src_gallery = src_root / "bounding_box_test_less"

    train_paths = list_images(src_train)
    query_paths = list_images(src_query)
    gallery_paths = list_images(src_gallery)

    selected_query, selected_gallery = choose_eval_subset(
        query_paths, gallery_paths, query_target, gallery_target, rng
    )
    selected_train = choose_train_subset(train_paths, train_target, rng)

    if dst_root.exists() and overwrite and not dry_run:
        shutil.rmtree(dst_root)
    elif dst_root.exists() and any(dst_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{dst_root} already exists and is not empty. Set OVERWRITE=1 to rebuild it."
        )

    split_plan = [
        (
            selected_train,
            "bounding_box_train_less",
            "bounding_box_train_tiny",
            src_root / "bounding_box_train_less_pose",
            dst_root / "bounding_box_train_tiny_pose",
        ),
        (
            selected_query,
            "query_less",
            "query_tiny",
            src_root / "query_pose_less",
            dst_root / "query_pose_tiny",
        ),
        (
            selected_gallery,
            "bounding_box_test_less",
            "bounding_box_test_tiny",
            src_root / "bounding_box_test_less_pose",
            dst_root / "bounding_box_test_tiny_pose",
        ),
    ]

    for paths, src_split, dst_split, pose_src, pose_dst in split_plan:
        copy_images(paths, src_root / src_split, dst_root / dst_split, dry_run)
        copy_pose(paths, pose_src, pose_dst, dry_run)
        copy_masks(paths, src_root / "sam3_part_masks", src_split, dst_split, dst_root, dry_run)

    selected_names = [path.name for path in selected_train + selected_query + selected_gallery]
    filter_color_json(
        src_root / "clothing_colors_less_nl.json",
        dst_root / "clothing_colors_tiny_nl.json",
        selected_names,
        dry_run,
    )

    summary = {
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "seed": seed,
        "train": len(selected_train),
        "query": len(selected_query),
        "gallery": len(selected_gallery),
        "query_plus_gallery": len(selected_query) + len(selected_gallery),
        "query_ids": len({pid_of(path) for path in selected_query}),
        "gallery_ids": len({pid_of(path) for path in selected_gallery}),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not dry_run:
        (dst_root / "tiny_manifest.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
PY
