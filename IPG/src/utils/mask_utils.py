import math
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


MaskInput = Union[torch.Tensor, Image.Image, Sequence[Image.Image]]


def build_soft_half_masks(
    height: int,
    width: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    transition_ratio: float = 0.08,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return smooth upper/lower fallback masks with shape [1, H, W]."""
    y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype).view(1, height, 1)
    softness = max(float(transition_ratio), 1e-4)
    upper = torch.sigmoid((0.5 - y) / softness).expand(1, height, width)
    lower = torch.sigmoid((y - 0.5) / softness).expand(1, height, width)
    return upper.clamp(0, 1), lower.clamp(0, 1)


def smooth_mask(mask: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    if kernel_size <= 1:
        return mask.clamp(0, 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    squeeze = mask.ndim == 3
    if squeeze:
        mask = mask.unsqueeze(0)
    mask = F.avg_pool2d(mask.float(), kernel_size, stride=1, padding=kernel_size // 2)
    mask = mask.clamp(0, 1)
    return mask.squeeze(0) if squeeze else mask


def pose_to_part_masks(
    pose_tensor: torch.Tensor,
    *,
    threshold: float = 0.05,
    transition_ratio: float = 0.08,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build coarse target part masks from a pose image tensor.

    The pose image is expected in [C, H, W] or [B, C, H, W] format and in [0, 1]
    or [-1, 1]. This is a safe fallback for missing parsing masks.
    """
    squeeze = pose_tensor.ndim == 3
    if squeeze:
        pose_tensor = pose_tensor.unsqueeze(0)
    if pose_tensor.ndim != 4:
        raise ValueError(f"Expected pose tensor with 3 or 4 dims, got {pose_tensor.ndim}.")

    pose = pose_tensor.float()
    if pose.min() < 0:
        pose = (pose + 1.0) / 2.0
    body = pose.abs().amax(dim=1, keepdim=True)
    body = (body > threshold).to(pose.dtype)
    body = smooth_mask(body, kernel_size=9)

    batch_size, _, height, width = body.shape
    upper, lower = build_soft_half_masks(
        height,
        width,
        device=pose.device,
        dtype=pose.dtype,
        transition_ratio=transition_ratio,
    )
    upper = upper.unsqueeze(0).expand(batch_size, -1, -1, -1)
    lower = lower.unsqueeze(0).expand(batch_size, -1, -1, -1)

    has_body = body.flatten(1).amax(dim=1).view(batch_size, 1, 1, 1)
    upper_mask = torch.where(has_body > 0, smooth_mask(body * upper, 9), upper)
    lower_mask = torch.where(has_body > 0, smooth_mask(body * lower, 9), lower)
    if squeeze:
        return upper_mask.squeeze(0), lower_mask.squeeze(0)
    return upper_mask, lower_mask


def pil_mask_to_tensor(
    image: Image.Image,
    height: int,
    width: int,
    *,
    flip: bool = False,
) -> torch.Tensor:
    image = image.convert("L")
    image = TF.resize(image, [height, width], interpolation=InterpolationMode.BILINEAR)
    if flip:
        image = TF.hflip(image)
    return TF.to_tensor(image).clamp(0, 1)


def load_mask_tensor(
    path: Optional[Union[str, Path]],
    height: int,
    width: int,
    *,
    flip: bool = False,
) -> Optional[torch.Tensor]:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    with Image.open(path) as image:
        return pil_mask_to_tensor(image, height, width, flip=flip)


def make_batch_half_masks(
    batch_size: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    upper, lower = build_soft_half_masks(height, width, device=device, dtype=dtype)
    return (
        upper.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous(),
        lower.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous(),
    )


def prepare_mask_batch(
    mask: Optional[MaskInput],
    batch_size: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    fallback: str = "upper",
) -> torch.Tensor:
    if mask is None:
        upper, lower = make_batch_half_masks(
            batch_size, height, width, device=device, dtype=dtype
        )
        return upper if fallback == "upper" else lower

    if isinstance(mask, torch.Tensor):
        tensor = mask
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError(f"Expected mask tensor with 3 or 4 dims, got {tensor.ndim}.")
        if tensor.shape[1] != 1:
            tensor = tensor[:, :1]
        tensor = tensor.to(device=device, dtype=dtype)
        tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    else:
        images = list(mask) if isinstance(mask, (list, tuple)) else [mask]
        tensors = [pil_mask_to_tensor(image, height, width) for image in images]
        tensor = torch.stack(tensors).to(device=device, dtype=dtype)

    if tensor.shape[0] == 1 and batch_size > 1:
        tensor = tensor.expand(batch_size, -1, -1, -1)
    elif tensor.shape[0] != batch_size:
        if batch_size % tensor.shape[0] != 0:
            raise ValueError(
                f"Cannot align mask batch {tensor.shape[0]} to requested batch {batch_size}."
            )
        tensor = tensor.repeat(batch_size // tensor.shape[0], 1, 1, 1)
    return tensor.clamp(0, 1)


def infer_token_hw(token_count: int, aspect_ratio: float) -> Tuple[int, int]:
    if token_count <= 0:
        raise ValueError("token_count must be positive.")
    height = max(1, int(round(math.sqrt(token_count * aspect_ratio))))
    while height > 1 and token_count % height != 0:
        height -= 1
    width = token_count // height
    if height * width != token_count:
        width = max(1, int(round(math.sqrt(token_count / max(aspect_ratio, 1e-6)))))
        height = max(1, token_count // width)
    return height, width


def mask_to_token_weights(
    mask: Optional[torch.Tensor],
    token_count: int,
    *,
    hidden_batch: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    if mask.ndim != 4:
        raise ValueError(f"Expected mask with 3 or 4 dims, got {mask.ndim}.")
    if mask.shape[1] != 1:
        mask = mask[:, :1]

    mask = mask.to(device=device, dtype=dtype)
    if mask.shape[0] != hidden_batch:
        if hidden_batch % mask.shape[0] != 0:
            return None
        mask = mask.repeat(hidden_batch // mask.shape[0], 1, 1, 1)

    aspect_ratio = float(mask.shape[-2]) / float(mask.shape[-1])
    layer_h, layer_w = infer_token_hw(token_count, aspect_ratio)
    resized = F.interpolate(mask, size=(layer_h, layer_w), mode="bilinear", align_corners=False)
    return resized.flatten(2).transpose(1, 2).contiguous().clamp(0, 1)
