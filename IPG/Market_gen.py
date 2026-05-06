"""
用于生成market1501的query和bounding_box_test
"""
import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils import check_min_version
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from reidmodel.trainsreid import make_model as make_transreid_model
from src.models.color_condition import ColorTextEncoder
from src.models.ifr import IFR
from src.models.mutual_self_attention import PartBankFusion
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline import Pose2ImagePipeline
from src.utils.util import import_filename, seed_everything


warnings.filterwarnings("ignore")
check_min_version("0.10.0.dev0")


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate pose-guided Market1501 images for query and bound_box_test."
    )
    parser.add_argument("--query_dir", type=str, default="/root/autodl-fs/datasets/market1501/query")
    parser.add_argument("--bound_box_test_dir", type=str, default="/root/autodl-fs/datasets/market1501/bounding_box_test")
    parser.add_argument("--ckpt_dir", type=str, default="/root/autodl-fs/epoch-10000/checkpoint-10000")
    parser.add_argument("--pose_dir", type=str, default="standard_poses")
    parser.add_argument("--config", type=str, default="./configs/inference.yaml")
    parser.add_argument("--reid_cfg_path", type=str, default="./cfg_transreid.pkl")
    parser.add_argument("--reid_ckpt_name", type=str, default="transformer_20.pth")
    parser.add_argument("--num_inference_steps", type=int, default=12)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--color_json", type=str, default=None)
    parser.add_argument("--mask_root", type=str, default=None)
    parser.add_argument("--disable_part_bank", action="store_true")
    parser.add_argument("--disable_color_structure", action="store_true")
    # Path 2 / 3 / 4 overrides (negative sigma -> disable timestep gate).
    parser.add_argument("--color_timestep_max_sigma", type=float, default=None)
    parser.add_argument("--color_min_layer_weight", type=float, default=None)
    parser.add_argument("--color_query_threshold", type=float, default=None)
    parser.add_argument(
        "--color_hard_gating",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Override hard query gating from config; 'true'/'false'.",
    )
    return parser.parse_args()


def load_config(config_path: str):
    if config_path.endswith(".yaml"):
        return OmegaConf.load(config_path)
    if config_path.endswith(".py"):
        return import_filename(config_path).cfg
    raise ValueError(f"Do not support this format config file: {config_path}")


def cfg_select(cfg, key: str, default=None):
    value = OmegaConf.select(cfg, key)
    return default if value is None else value


def get_weight_dtype(weight_dtype_name: str):
    if weight_dtype_name == "fp16":
        return torch.float16
    if weight_dtype_name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported weight dtype: {weight_dtype_name}")


def build_scheduler(cfg):
    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs, resolve=True)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    return DDIMScheduler(**sched_kwargs)


def build_reid_transform():
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return transforms.Compose(
        [
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            normalize,
        ]
    )


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def sort_pose_key(path: Path):
    stem = path.stem
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def collect_pose_images(pose_dir: Path) -> List[Tuple[str, Image.Image, Path]]:
    pose_paths = sorted([path for path in pose_dir.iterdir() if is_image_file(path)], key=sort_pose_key)
    if not pose_paths:
        raise RuntimeError(f"No pose images found in {pose_dir}")

    pose_images = []
    for pose_index, pose_path in enumerate(pose_paths, start=1):
        pose_name = f"pose{pose_index}"
        with Image.open(pose_path) as pose_image:
            pose_images.append((pose_name, pose_image.convert("RGB"), pose_path))
    return pose_images


def collect_ref_images(split_dir: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in split_dir.iterdir()
            if is_image_file(path) and not path.name.startswith("0000")
        ]
    )


def load_color_descriptions(path: str):
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        print(f"[WARN] Color json not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def get_color_texts(color_descriptions: dict, image_path: Path, num_poses: int):
    record = color_descriptions.get(image_path.name) or color_descriptions.get(image_path.stem)
    if not isinstance(record, dict):
        return [""] * num_poses, [""] * num_poses
    return (
        [str(record.get("upper", "") or "")] * num_poses,
        [str(record.get("lower", "") or "")] * num_poses,
    )


def resolve_mask_file(mask_root: Path, image_path: Path, part: str, group_name: str = None):
    roots = [mask_root]
    if group_name:
        roots.insert(0, mask_root / group_name)
    candidates = [
        root / part / image_path.name
        for root in roots
    ] + [
        root / f"{image_path.stem}_{part}{image_path.suffix}"
        for root in roots
    ] + [
        root / image_path.name
        for root in roots
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
        stem = candidate.with_suffix("")
        for ext in IMAGE_EXTENSIONS:
            ext_candidate = stem.with_suffix(ext)
            if ext_candidate.exists():
                return ext_candidate
    return None


def load_mask_pair(mask_root: str, image_path: Path, group_name: str = None):
    if not mask_root:
        return None, None
    root = Path(mask_root)
    upper_path = resolve_mask_file(root, image_path, "upper", group_name=group_name)
    lower_path = resolve_mask_file(root, image_path, "lower", group_name=group_name)
    if upper_path is None or lower_path is None:
        return None, None
    return Image.open(upper_path).convert("L"), Image.open(lower_path).convert("L")


def load_models(cfg, args, device: torch.device, part_bank_enabled: bool, color_enabled: bool):
    scheduler = build_scheduler(cfg)

    # Keep inference in float32 to match the existing inference.py behavior and
    # avoid dtype mismatches inside the reference UNet / time embedding blocks.
    inference_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        device=device, dtype=inference_dtype
    )
    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device=device, dtype=inference_dtype)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
        },
    ).to(device=device, dtype=inference_dtype)
    pose_guider = PoseGuider(conditioning_embedding_channels=320).to(
        device=device, dtype=inference_dtype
    )
    ifr = IFR().to(device=device, dtype=inference_dtype)
    part_bank_fusion = PartBankFusion(
        lambda_init=cfg_select(cfg, "features.part_reference_bank.lambda_init", 0.0)
    ).to(device=device, dtype=inference_dtype)

    ckpt_dir = Path(args.ckpt_dir)
    reference_unet.load_state_dict(
        torch.load(ckpt_dir / "reference_unet.pth", map_location="cpu"),
        strict=True,
    )
    denoising_unet.load_state_dict(
        torch.load(ckpt_dir / "denoising_unet.pth", map_location="cpu"),
        strict=True,
    )
    pose_guider.load_state_dict(
        torch.load(ckpt_dir / "pose_guider.pth", map_location="cpu"),
        strict=True,
    )
    ifr.load_state_dict(
        torch.load(ckpt_dir / "IFR.pth", map_location="cpu"),
        strict=True,
    )
    part_fusion_path = ckpt_dir / "part_bank_fusion.pth"
    if part_fusion_path.exists():
        part_bank_fusion.load_state_dict(
            torch.load(part_fusion_path, map_location="cpu"), strict=True
        )
    elif part_bank_enabled:
        print(
            f"[WARN] {part_fusion_path} not found; using lambda_init for part bank fusion."
        )

    with open(args.reid_cfg_path, "rb") as handle:
        reid_cfg = pickle.load(handle)
    reid_net = make_transreid_model(reid_cfg, num_class=751, camera_num=0, view_num=1)
    reid_net.load_param(str(ckpt_dir / args.reid_ckpt_name))
    reid_net.to(device=device, dtype=inference_dtype)

    pipe = Pose2ImagePipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
        part_bank_fusion=part_bank_fusion,
    ).to(device)
    color_encoder = None
    if color_enabled:
        color_encoder = ColorTextEncoder(
            cfg_select(cfg, "features.color_structure.clip_model_path", cfg.base_model_path),
            tokenizer_subfolder=cfg_select(
                cfg, "features.color_structure.tokenizer_subfolder", "tokenizer"
            ),
            text_encoder_subfolder=cfg_select(
                cfg, "features.color_structure.text_encoder_subfolder", "text_encoder"
            ),
        ).to(device=device, dtype=inference_dtype)
        color_encoder.eval()

    for model in (pipe.vae, pipe.reference_unet, pipe.denoising_unet, pipe.pose_guider, ifr, reid_net, part_bank_fusion):
        model.eval()
        model.requires_grad_(False)
    if color_encoder is not None:
        color_encoder.requires_grad_(False)

    return pipe, ifr, reid_net, color_encoder


def build_identity_embeddings(
    reid_net,
    ifr,
    reid_tensor: torch.Tensor,
    num_poses: int,
    device: torch.device,
):
    pose_batch = reid_tensor.unsqueeze(0).repeat(num_poses, 1, 1, 1).to(device=device)
    zeros_batch = torch.zeros_like(pose_batch, device=device)
    reid_batch = torch.cat([zeros_batch, pose_batch], dim=0)
    cam_label = torch.zeros(reid_batch.shape[0], dtype=torch.long, device=device)
    view_label = torch.ones(reid_batch.shape[0], dtype=torch.long, device=device)
    reid_features = reid_net(reid_batch, cam_label=cam_label, view_label=view_label)
    return ifr(reid_features)


def ensure_output_dirs(output_root: Path, pose_names: Sequence[str]):
    output_root.mkdir(parents=True, exist_ok=True)
    for pose_name in pose_names:
        (output_root / pose_name).mkdir(parents=True, exist_ok=True)


def save_generated_images(
    generated_images: torch.Tensor,
    pose_names: Sequence[str],
    output_root: Path,
    file_name: str,
):
    to_pil = transforms.ToPILImage()
    for pose_index, pose_name in enumerate(pose_names):
        generated = generated_images[pose_index, :, 0].cpu()
        result_image = to_pil(generated).resize((128, 256), Image.Resampling.BILINEAR)
        result_image.save(output_root / pose_name / file_name)


def generate_split(
    split_dir: Path,
    pipe,
    ifr,
    reid_net,
    color_encoder,
    color_descriptions: dict,
    reid_transform,
    pose_images: Sequence[Tuple[str, Image.Image, Path]],
    cfg,
    args,
    generator: torch.Generator,
    device: torch.device,
    part_bank_enabled: bool,
    color_enabled: bool,
    color_gating: dict,
):
    if not split_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {split_dir}")

    output_root = split_dir.parent / f"{split_dir.name}_gen"
    pose_names = [pose_name for pose_name, _, _ in pose_images]
    pose_pils = [pose_image.copy() for _, pose_image, _ in pose_images]
    pose_paths = [pose_path for _, _, pose_path in pose_images]
    ensure_output_dirs(output_root, pose_names)

    image_paths = collect_ref_images(split_dir)
    print(f"Processing {split_dir} -> {output_root}, valid images: {len(image_paths)}")

    for image_index, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as ref_handle:
            ref_image = ref_handle.convert("RGB")
        reid_input = reid_transform(ref_image).to(device=device, dtype=torch.float32)
        feature_embeds = build_identity_embeddings(
            reid_net=reid_net,
            ifr=ifr,
            reid_tensor=reid_input,
            num_poses=len(pose_images),
            device=device,
        )
        color_upper_states = None
        color_lower_states = None
        if color_enabled and color_encoder is not None:
            upper_texts, lower_texts = get_color_texts(
                color_descriptions, image_path, len(pose_images)
            )
            color_upper_states, color_lower_states = color_encoder.encode_pair(
                upper_texts,
                lower_texts,
                device=device,
                dtype=feature_embeds.dtype,
            )
        ref_upper_mask, ref_lower_mask = load_mask_pair(
            args.mask_root, image_path, group_name=split_dir.name
        )
        target_upper_masks = []
        target_lower_masks = []
        if args.mask_root:
            for pose_path in pose_paths:
                upper_mask, lower_mask = load_mask_pair(
                    args.mask_root, pose_path, group_name="standard_poses"
                )
                if upper_mask is None or lower_mask is None:
                    target_upper_masks = []
                    target_lower_masks = []
                    break
                target_upper_masks.append(upper_mask)
                target_lower_masks.append(lower_mask)
        generated_images = pipe(
            feature_embeds,
            [ref_image.copy() for _ in pose_images],
            pose_pils,
            cfg.data.train_height,
            cfg.data.train_width,
            args.num_inference_steps,
            args.guidance_scale,
            batch_size=len(pose_images),
            generator=generator,
            ref_upper_mask=[ref_upper_mask.copy() for _ in pose_images]
            if ref_upper_mask is not None
            else None,
            ref_lower_mask=[ref_lower_mask.copy() for _ in pose_images]
            if ref_lower_mask is not None
            else None,
            target_upper_mask=target_upper_masks if target_upper_masks else None,
            target_lower_mask=target_lower_masks if target_lower_masks else None,
            color_upper_states=color_upper_states,
            color_lower_states=color_lower_states,
            part_bank_enabled=part_bank_enabled,
            color_structure_enabled=color_enabled,
            color_scale=cfg_select(cfg, "features.color_structure.color_scale", 0.25),
            color_timestep_max_sigma=color_gating["timestep_max_sigma"],
            color_min_layer_weight=color_gating["min_layer_weight"],
            color_hard_gating=color_gating["hard_gating"],
            color_query_threshold=color_gating["query_threshold"],
        ).images
        save_generated_images(generated_images, pose_names, output_root, image_path.name)
        print(f"[{image_index}/{len(image_paths)}] Saved {image_path.name}")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Market1501 generation.")

    device = torch.device("cuda")
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    part_bank_enabled = bool(
        cfg_select(cfg, "features.part_reference_bank.enabled", False)
    ) and not args.disable_part_bank
    color_enabled = bool(
        cfg_select(cfg, "features.color_structure.enabled", False)
    ) and not args.disable_color_structure
    color_json_path = args.color_json or cfg_select(
        cfg, "features.color_structure.color_json_path"
    )
    color_descriptions = load_color_descriptions(color_json_path) if color_enabled else {}
    if args.mask_root is None:
        args.mask_root = cfg_select(
            cfg, "features.part_reference_bank.masks.ref_upper_root"
        )

    # Resolve color gating knobs (Path 2 / 3 / 4). Negative sigma in CLI/yaml
    # disables the timestep gate; CLI overrides win over the yaml.
    cfg_max_sigma = cfg_select(cfg, "features.color_structure.timestep_max_sigma")
    if args.color_timestep_max_sigma is not None:
        cfg_max_sigma = args.color_timestep_max_sigma
    color_timestep_max_sigma = (
        float(cfg_max_sigma) if cfg_max_sigma is not None and float(cfg_max_sigma) >= 0 else None
    )
    color_min_layer_weight = float(
        args.color_min_layer_weight
        if args.color_min_layer_weight is not None
        else cfg_select(cfg, "features.color_structure.min_layer_weight", 0.0)
    )
    color_query_threshold = float(
        args.color_query_threshold
        if args.color_query_threshold is not None
        else cfg_select(cfg, "features.color_structure.query_threshold", 0.5)
    )
    if args.color_hard_gating is not None:
        color_hard_gating = args.color_hard_gating == "true"
    else:
        color_hard_gating = bool(
            cfg_select(cfg, "features.color_structure.hard_query_gating", False)
        )
    color_gating = {
        "timestep_max_sigma": color_timestep_max_sigma,
        "min_layer_weight": color_min_layer_weight,
        "hard_gating": color_hard_gating,
        "query_threshold": color_query_threshold,
    }

    pipe, ifr, reid_net, color_encoder = load_models(
        cfg, args, device, part_bank_enabled, color_enabled
    )
    reid_transform = build_reid_transform()
    pose_images = collect_pose_images(Path(args.pose_dir))
    generator = torch.Generator(device=device)
    if cfg.seed is not None:
        generator.manual_seed(cfg.seed)

    with torch.inference_mode():
        generate_split(
            split_dir=Path(args.query_dir),
            pipe=pipe,
            ifr=ifr,
            reid_net=reid_net,
            color_encoder=color_encoder,
            color_descriptions=color_descriptions,
            reid_transform=reid_transform,
            pose_images=pose_images,
            cfg=cfg,
            args=args,
            generator=generator,
            device=device,
            part_bank_enabled=part_bank_enabled,
            color_enabled=color_enabled,
            color_gating=color_gating,
        )
        generate_split(
            split_dir=Path(args.bound_box_test_dir),
            pipe=pipe,
            ifr=ifr,
            reid_net=reid_net,
            color_encoder=color_encoder,
            color_descriptions=color_descriptions,
            reid_transform=reid_transform,
            pose_images=pose_images,
            cfg=cfg,
            args=args,
            generator=generator,
            device=device,
            part_bank_enabled=part_bank_enabled,
            color_enabled=color_enabled,
            color_gating=color_gating,
        )


if __name__ == "__main__":
    main()
