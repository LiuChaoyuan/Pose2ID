import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import RandomErasing
from torchvision.transforms import functional as TF


DEFAULT_IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
)


def _is_image_file(path: Path, image_extensions: Sequence[str]) -> bool:
    return path.is_file() and path.suffix.lower() in image_extensions


def _parse_identity(
    path: Path,
    root: Path,
    identity_mode: str,
    filename_delimiter: str,
    filename_index: int,
) -> str:
    if identity_mode == "parent":
        rel_parts = path.relative_to(root).parts
        if len(rel_parts) < 2:
            raise ValueError(
                f"Expected identity folders under {root}, but found flat image path: {path}"
            )
        return rel_parts[0]

    if identity_mode == "filename":
        parts = path.stem.split(filename_delimiter)
        if not parts:
            raise ValueError(f"Unable to parse identity from filename: {path.name}")
        index = filename_index if filename_index >= 0 else filename_index + len(parts)
        if index < 0 or index >= len(parts):
            raise ValueError(
                f"filename_index={filename_index} is out of range for filename {path.name}"
            )
        return parts[index]

    raise ValueError(f"Unsupported identity_mode: {identity_mode}")


def _collect_grouped_images(
    root: Path,
    image_extensions: Sequence[str],
    identity_mode: str,
    filename_delimiter: str,
    filename_index: int,
) -> Dict[str, List[Path]]:
    grouped: Dict[str, List[Path]] = {}
    for path in sorted(root.rglob("*")):
        if not _is_image_file(path, image_extensions):
            continue
        identity = _parse_identity(
            path=path,
            root=root,
            identity_mode=identity_mode,
            filename_delimiter=filename_delimiter,
            filename_index=filename_index,
        )
        grouped.setdefault(identity, []).append(path)
    return grouped


def _resolve_pose_path(
    target_path: Path,
    target_root: Path,
    pose_root: Path,
    image_extensions: Sequence[str],
) -> Optional[Path]:
    rel_path = target_path.relative_to(target_root)
    direct_candidate = pose_root / rel_path
    if direct_candidate.exists():
        return direct_candidate

    stem_candidate = direct_candidate.with_suffix("")
    for ext in image_extensions:
        candidate = stem_candidate.with_suffix(ext)
        if candidate.exists():
            return candidate

    return None


class IPGDataset(Dataset):
    def __init__(
        self,
        dataset_specs: Sequence[dict],
        image_height: int,
        image_width: int,
        reid_height: int,
        reid_width: int,
        image_extensions: Optional[Sequence[str]] = None,
        ref_random_flip: bool = True,
        ref_random_erasing_prob: float = 0.0,
        ref_random_erasing_on: str = "reid",
        allow_same_reference: bool = False,
    ):
        self.dataset_specs = list(dataset_specs)
        self.image_height = image_height
        self.image_width = image_width
        self.reid_height = reid_height
        self.reid_width = reid_width
        self.image_extensions = tuple(
            ext.lower() for ext in (image_extensions or DEFAULT_IMAGE_EXTENSIONS)
        )
        self.ref_random_flip = ref_random_flip
        self.ref_random_erasing_prob = ref_random_erasing_prob
        self.ref_random_erasing_on = ref_random_erasing_on
        self.allow_same_reference = allow_same_reference
        self.random_erasing = RandomErasing(
            p=ref_random_erasing_prob,
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3),
            value="random",
        )

        self.identity_to_refs: Dict[Tuple[str, str], List[Path]] = {}
        self.target_records: List[dict] = []
        self.dataset_summaries: List[dict] = []
        self._build_index()

        if not self.target_records:
            raise RuntimeError("No valid IPG training samples were found.")

    def _build_index(self) -> None:
        for raw_spec in self.dataset_specs:
            spec = dict(raw_spec)
            dataset_name = spec.get("name", "default")
            identity_mode = spec.get("identity_mode", "parent")
            filename_delimiter = spec.get("filename_delimiter", "_")
            filename_index = int(spec.get("filename_index", 0))

            ref_root_value = (
                spec.get("ref_root")
                or spec.get("image_root")
                or spec.get("target_root")
            )
            target_root_value = (
                spec.get("target_root")
                or spec.get("image_root")
                or spec.get("ref_root")
            )
            pose_root_value = spec.get("pose_root")

            if ref_root_value is None or target_root_value is None or pose_root_value is None:
                raise ValueError(
                    f"Dataset spec '{dataset_name}' must provide pose_root and at least one of "
                    "image_root/ref_root/target_root."
                )

            ref_root = Path(ref_root_value)
            target_root = Path(target_root_value)
            pose_root = Path(pose_root_value)

            if not ref_root.exists():
                raise FileNotFoundError(f"ref_root does not exist: {ref_root}")
            if not target_root.exists():
                raise FileNotFoundError(f"target_root does not exist: {target_root}")
            if not pose_root.exists():
                raise FileNotFoundError(f"pose_root does not exist: {pose_root}")

            ref_grouped = _collect_grouped_images(
                root=ref_root,
                image_extensions=self.image_extensions,
                identity_mode=identity_mode,
                filename_delimiter=filename_delimiter,
                filename_index=filename_index,
            )
            target_grouped = _collect_grouped_images(
                root=target_root,
                image_extensions=self.image_extensions,
                identity_mode=identity_mode,
                filename_delimiter=filename_delimiter,
                filename_index=filename_index,
            )

            kept_targets = 0
            dropped_missing_pose = 0
            dropped_missing_ref = 0
            for identity, target_paths in target_grouped.items():
                ref_paths = ref_grouped.get(identity, [])
                if not ref_paths:
                    dropped_missing_ref += len(target_paths)
                    continue

                identity_key = (dataset_name, identity)
                self.identity_to_refs[identity_key] = ref_paths

                for target_path in target_paths:
                    pose_path = _resolve_pose_path(
                        target_path=target_path,
                        target_root=target_root,
                        pose_root=pose_root,
                        image_extensions=self.image_extensions,
                    )
                    if pose_path is None:
                        dropped_missing_pose += 1
                        continue

                    self.target_records.append(
                        {
                            "dataset_name": dataset_name,
                            "identity": identity,
                            "target_path": target_path,
                            "pose_path": pose_path,
                            "identity_key": identity_key,
                        }
                    )
                    kept_targets += 1

            self.dataset_summaries.append(
                {
                    "name": dataset_name,
                    "num_identities": len(ref_grouped),
                    "num_targets": kept_targets,
                    "dropped_missing_ref": dropped_missing_ref,
                    "dropped_missing_pose": dropped_missing_pose,
                }
            )

    def __len__(self) -> int:
        return len(self.target_records)

    def _load_rgb(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _maybe_flip(self, image: Image.Image) -> Image.Image:
        if self.ref_random_flip and random.random() < 0.5:
            return TF.hflip(image)
        return image

    def _build_diffusion_tensor(self, image: Image.Image) -> torch.Tensor:
        image = TF.resize(
            image,
            size=[self.image_height, self.image_width],
            interpolation=InterpolationMode.BILINEAR,
        )
        tensor = TF.to_tensor(image)
        return TF.normalize(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def _build_reid_tensor(self, image: Image.Image) -> torch.Tensor:
        image = TF.resize(
            image,
            size=[self.reid_height, self.reid_width],
            interpolation=InterpolationMode.BILINEAR,
        )
        tensor = TF.to_tensor(image)
        return TF.normalize(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def _build_pose_tensor(self, image: Image.Image) -> torch.Tensor:
        image = TF.resize(
            image,
            size=[self.image_height, self.image_width],
            interpolation=InterpolationMode.BILINEAR,
        )
        return TF.to_tensor(image)

    def __getitem__(self, index: int) -> dict:
        record = self.target_records[index]
        ref_candidates = self.identity_to_refs[record["identity_key"]]

        if self.allow_same_reference:
            valid_refs = ref_candidates
        else:
            valid_refs = [path for path in ref_candidates if path != record["target_path"]]
            if not valid_refs:
                valid_refs = ref_candidates

        ref_path = random.choice(valid_refs)
        ref_image = self._maybe_flip(self._load_rgb(ref_path))
        target_image = self._load_rgb(record["target_path"])
        pose_image = self._load_rgb(record["pose_path"])

        ref_tensor = self._build_diffusion_tensor(ref_image)
        reid_tensor = self._build_reid_tensor(ref_image)
        if self.ref_random_erasing_on in {"reid", "both"} and self.ref_random_erasing_prob > 0:
            reid_tensor = self.random_erasing(reid_tensor)
        if self.ref_random_erasing_on == "both" and self.ref_random_erasing_prob > 0:
            ref_tensor = self.random_erasing(ref_tensor)

        target_tensor = self._build_diffusion_tensor(target_image)
        pose_tensor = self._build_pose_tensor(pose_image)

        return {
            "ref_image": ref_tensor,
            "target_image": target_tensor,
            "pose_image": pose_tensor,
            "reid_image": reid_tensor,
            "dataset_name": record["dataset_name"],
            "identity": record["identity"],
            "ref_path": str(ref_path),
            "target_path": str(record["target_path"]),
            "pose_path": str(record["pose_path"]),
        }
