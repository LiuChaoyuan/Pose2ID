import argparse
import math
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from reidmodel.trainsreid import make_model as make_transreid_model
from src.models.color_condition import ColorTextEncoder
from src.ipg_dataset import IPGDataset
from src.models.ifr import IFR
from src.models.mutual_self_attention import PartBankFusion, ReferenceAttentionControl
from src.models.pose_loss import DifferentiablePoseLoss
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline import Pose2ImagePipeline
from src.utils.util import delete_additional_ckpt, import_filename


logger = get_logger(__name__)


class IPGTrainModel(nn.Module):
    def __init__(
        self,
        ifr: IFR,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider: PoseGuider,
        part_bank_enabled: bool = False,
        part_bank_lambda_init: float = 0.0,
        color_structure_enabled: bool = False,
        color_scale: float = 0.0,
        color_timestep_max_sigma: Optional[float] = None,
        color_min_layer_weight: float = 0.0,
        color_hard_gating: bool = False,
        color_query_threshold: float = 0.5,
        noise_scheduler=None,
    ):
        super().__init__()
        self.ifr = ifr
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.part_bank_enabled = part_bank_enabled
        self.color_structure_enabled = color_structure_enabled
        self.color_scale = float(color_scale)
        self.part_bank_fusion = PartBankFusion(lambda_init=part_bank_lambda_init)
        self.reference_control_writer = ReferenceAttentionControl(
            reference_unet,
            do_classifier_free_guidance=False,
            mode="write",
            fusion_blocks="full",
            part_bank_enabled=part_bank_enabled,
        )
        self.reference_control_reader = ReferenceAttentionControl(
            denoising_unet,
            do_classifier_free_guidance=False,
            mode="read",
            fusion_blocks="full",
            part_bank_enabled=part_bank_enabled,
            part_bank_fusion=self.part_bank_fusion,
            color_structure_enabled=color_structure_enabled,
            color_scale=color_scale,
            color_timestep_max_sigma=color_timestep_max_sigma,
            color_min_layer_weight=color_min_layer_weight,
            color_hard_gating=color_hard_gating,
            color_query_threshold=color_query_threshold,
        )
        # Buffer the diffusion noise schedule so the training loop can derive a
        # per-step sigma to drive the color sigma gate (Path 2). We pick the
        # batch mean sigma since attention hooks expect a single scalar.
        if noise_scheduler is not None:
            alphas_cumprod = noise_scheduler.alphas_cumprod.clone().float()
        else:
            alphas_cumprod = torch.zeros(0)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        ref_image_latents: torch.Tensor,
        reid_features: torch.Tensor,
        pose_images: torch.Tensor,
        ref_upper_latents: Optional[torch.Tensor] = None,
        ref_lower_latents: Optional[torch.Tensor] = None,
        target_upper_mask: Optional[torch.Tensor] = None,
        target_lower_mask: Optional[torch.Tensor] = None,
        color_upper_states: Optional[torch.Tensor] = None,
        color_lower_states: Optional[torch.Tensor] = None,
        cond_dropout_prob: float = 0.0,
    ) -> torch.Tensor:
        identity_embeds = self.ifr(reid_features)

        if cond_dropout_prob > 0:
            keep_mask = (
                torch.rand(identity_embeds.shape[0], device=identity_embeds.device)
                >= cond_dropout_prob
            )
            identity_embeds = identity_embeds * keep_mask[:, None, None].to(identity_embeds.dtype)
            ref_image_latents = ref_image_latents * keep_mask[:, None, None, None].to(
                ref_image_latents.dtype
            )
            if ref_upper_latents is not None:
                ref_upper_latents = ref_upper_latents * keep_mask[:, None, None, None].to(
                    ref_upper_latents.dtype
                )
            if ref_lower_latents is not None:
                ref_lower_latents = ref_lower_latents * keep_mask[:, None, None, None].to(
                    ref_lower_latents.dtype
                )
            if color_upper_states is not None:
                color_upper_states = color_upper_states * keep_mask[:, None, None].to(
                    color_upper_states.dtype
                )
            if color_lower_states is not None:
                color_lower_states = color_lower_states * keep_mask[:, None, None].to(
                    color_lower_states.dtype
                )

        pose_cond_tensor = pose_images.unsqueeze(2)
        pose_fea = self.pose_guider(pose_cond_tensor)

        try:
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_control_reader.set_target_masks(
                target_upper_mask, target_lower_mask
            )
            if self.color_structure_enabled:
                self.reference_control_reader.set_color_states(
                    color_upper_states,
                    color_lower_states,
                    color_scale=self.color_scale,
                )
                # Path 2: tell the controller the noise level of this batch so
                # the sigma gate from the yaml takes effect during training as
                # well, keeping train / inference behavior consistent.
                if self.alphas_cumprod.numel() > 0:
                    sigma_per_sample = (
                        1.0 - self.alphas_cumprod[timesteps].clamp(min=0)
                    ).sqrt()
                    self.reference_control_reader.set_active_sigma(
                        float(sigma_per_sample.mean().item())
                    )
            self.reference_control_writer.set_active_bank("global")
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=identity_embeds,
                return_dict=False,
            )
            if (
                self.part_bank_enabled
                and ref_upper_latents is not None
                and ref_lower_latents is not None
            ):
                self.reference_control_writer.set_active_bank("upper")
                self.reference_unet(
                    ref_upper_latents,
                    ref_timesteps,
                    encoder_hidden_states=identity_embeds,
                    return_dict=False,
                )
                self.reference_control_writer.set_active_bank("lower")
                self.reference_unet(
                    ref_lower_latents,
                    ref_timesteps,
                    encoder_hidden_states=identity_embeds,
                    return_dict=False,
                )
            self.reference_control_reader.update(self.reference_control_writer)
            model_pred = self.denoising_unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=identity_embeds,
                pose_cond_fea=pose_fea,
            ).sample
        finally:
            self.reference_control_reader.clear()
            self.reference_control_writer.clear()

        return model_pred


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train_ipg.yaml")
    return parser.parse_args()


def load_config(config_path: str):
    if config_path.endswith(".yaml"):
        return OmegaConf.load(config_path)
    if config_path.endswith(".py"):
        return import_filename(config_path).cfg
    raise ValueError(f"Unsupported config format: {config_path}")


def cfg_select(cfg, key: str, default=None):
    value = OmegaConf.select(cfg, key)
    return default if value is None else value


def resolve_train_sets(cfg) -> List[dict]:
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    if data_cfg.get("train_sets"):
        return data_cfg["train_sets"]
    return [
        {
            "name": data_cfg.get("name", "default"),
            "image_root": data_cfg.get("image_root"),
            "ref_root": data_cfg.get("ref_root"),
            "target_root": data_cfg.get("target_root"),
            "pose_root": data_cfg.get("pose_root"),
            "identity_mode": data_cfg.get("identity_mode", "parent"),
            "filename_delimiter": data_cfg.get("filename_delimiter", "_"),
            "filename_index": data_cfg.get("filename_index", 0),
        }
    ]


def get_weight_dtype(weight_dtype_name: str):
    if weight_dtype_name == "fp16":
        return torch.float16
    if weight_dtype_name == "fp32":
        return torch.float32
    if weight_dtype_name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported weight dtype: {weight_dtype_name}")


def compute_snr(noise_scheduler, timesteps):
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=timesteps.device)
    alpha = alphas_cumprod[timesteps] ** 0.5
    sigma = (1.0 - alphas_cumprod[timesteps]) ** 0.5
    return (alpha / sigma) ** 2


def load_reid_model(cfg, device: torch.device):
    reid_cfg_path = Path(cfg.reid.cfg_path)
    reid_ckpt_path = Path(cfg.reid.ckpt_path)
    if not reid_cfg_path.exists():
        raise FileNotFoundError(f"ReID config pickle not found: {reid_cfg_path}")
    if not reid_ckpt_path.exists():
        raise FileNotFoundError(f"ReID checkpoint not found: {reid_ckpt_path}")

    reid_cfg = pickle.load(open(reid_cfg_path, "rb"))
    reid_model = make_transreid_model(
        reid_cfg,
        num_class=cfg.reid.num_classes,
        camera_num=cfg.reid.camera_num,
        view_num=cfg.reid.view_num,
    )
    reid_model.load_param(str(reid_ckpt_path))
    reid_model.to(device=device, dtype=torch.float32)
    reid_model.eval()
    reid_model.requires_grad_(False)
    return reid_model


def maybe_enable_xformers(*models):
    if not is_xformers_available():
        raise ImportError(
            "xformers is not available, but enable_xformers_memory_efficient_attention is True."
        )

    for model in models:
        if hasattr(model, "enable_xformers_memory_efficient_attention"):
            model.enable_xformers_memory_efficient_attention()


def build_scheduler(cfg):
    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs, resolve=True)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    return DDPMScheduler(**sched_kwargs)


def build_validation_scheduler(cfg):
    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs, resolve=True)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    return DDIMScheduler(**sched_kwargs)


def save_component_weights(output_dir: Path, global_step: int, accelerator: Accelerator, ipg_model):
    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(checkpoint_dir))

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(ipg_model)
        torch.save(unwrapped.reference_unet.state_dict(), checkpoint_dir / "reference_unet.pth")
        torch.save(unwrapped.denoising_unet.state_dict(), checkpoint_dir / "denoising_unet.pth")
        torch.save(unwrapped.pose_guider.state_dict(), checkpoint_dir / "pose_guider.pth")
        torch.save(unwrapped.ifr.state_dict(), checkpoint_dir / "IFR.pth")
        if hasattr(unwrapped, "part_bank_fusion"):
            torch.save(
                unwrapped.part_bank_fusion.state_dict(),
                checkpoint_dir / "part_bank_fusion.pth",
            )
    return checkpoint_dir


def _sorted_checkpoint_dirs(output_dir: Path) -> List[Path]:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            int(path.name.split("-")[-1])
        except ValueError:
            continue
        checkpoints.append(path)
    return sorted(checkpoints, key=lambda p: int(p.name.split("-")[-1]))


def maybe_resume(accelerator: Accelerator, output_dir: Path, resume_from_checkpoint: Optional[str]):
    if not resume_from_checkpoint:
        return 0

    if resume_from_checkpoint == "latest":
        checkpoint_dirs = _sorted_checkpoint_dirs(output_dir)
        if not checkpoint_dirs:
            raise FileNotFoundError(f"No checkpoints found under {output_dir}")
        resume_path = checkpoint_dirs[-1]
    else:
        resume_path = Path(resume_from_checkpoint)
        if not resume_path.is_absolute():
            resume_path = output_dir / resume_path

    if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    accelerator.load_state(str(resume_path))
    logger.info(f"Resumed from checkpoint: {resume_path}")
    return int(resume_path.name.split("-")[-1])


def build_reid_condition(reid_model, reid_images: torch.Tensor, view_label: int = 1) -> torch.Tensor:
    batch_size = reid_images.shape[0]
    cam_labels = torch.zeros(batch_size, dtype=torch.long, device=reid_images.device)
    view_labels = torch.full((batch_size,), view_label, dtype=torch.long, device=reid_images.device)
    with torch.no_grad():
        features = reid_model(
            reid_images.to(dtype=torch.float32),
            cam_label=cam_labels,
            view_label=view_labels,
        )
    return features


def image_to_reid_tensor(image: Image.Image, height: int, width: int) -> torch.Tensor:
    image = TF.resize(image, [height, width], interpolation=InterpolationMode.BILINEAR)
    tensor = TF.to_tensor(image)
    return TF.normalize(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


def run_validation(
    cfg,
    accelerator: Accelerator,
    vae: AutoencoderKL,
    reid_model,
    ipg_model,
    output_dir: Path,
    global_step: int,
):
    if not getattr(cfg.validation, "enabled", False):
        return

    ref_dir = Path(cfg.validation.ref_dir)
    pose_dir = Path(cfg.validation.pose_dir)
    if not ref_dir.exists() or not pose_dir.exists():
        logger.warning("Validation skipped because validation ref_dir or pose_dir does not exist.")
        return

    ref_paths = sorted([p for p in ref_dir.iterdir() if p.is_file()])
    pose_paths = sorted([p for p in pose_dir.iterdir() if p.is_file()])
    if not ref_paths or not pose_paths:
        logger.warning("Validation skipped because validation images are empty.")
        return

    out_dir = output_dir / "validation" / f"step-{global_step}"
    out_dir.mkdir(parents=True, exist_ok=True)

    unwrapped = accelerator.unwrap_model(ipg_model)
    pipe = Pose2ImagePipeline(
        vae=vae,
        reference_unet=unwrapped.reference_unet,
        denoising_unet=unwrapped.denoising_unet,
        pose_guider=unwrapped.pose_guider,
        scheduler=build_validation_scheduler(cfg),
        part_bank_fusion=getattr(unwrapped, "part_bank_fusion", None),
    ).to(accelerator.device)

    generator = torch.Generator(device=accelerator.device)
    if cfg.seed is not None:
        generator.manual_seed(cfg.seed + global_step)

    with torch.no_grad():
        for ref_path in ref_paths:
            ref_image = Image.open(ref_path).convert("RGB")
            pose_images = [Image.open(path).convert("RGB") for path in pose_paths]
            ref_reid = image_to_reid_tensor(
                ref_image, height=cfg.data.reid_height, width=cfg.data.reid_width
            ).unsqueeze(0).to(accelerator.device)
            reid_inputs = torch.cat([torch.zeros_like(ref_reid), ref_reid], dim=0)
            reid_features = build_reid_condition(reid_model, reid_inputs)
            feature_embeds = unwrapped.ifr(
                reid_features.to(
                    device=accelerator.device,
                    dtype=unwrapped.ifr.proj_motion.weight.dtype,
                )
            )

            generated = pipe(
                feature_embeds,
                [ref_image for _ in pose_images],
                pose_images,
                cfg.data.train_height,
                cfg.data.train_width,
                cfg.validation.num_inference_steps,
                cfg.validation.guidance_scale,
                batch_size=len(pose_images),
                generator=generator,
                part_bank_enabled=bool(
                    cfg_select(cfg, "features.part_reference_bank.enabled", False)
                ),
            ).images

            preview_w = cfg.validation.preview_width
            preview_h = cfg.validation.preview_height
            canvas = Image.new(
                "RGB",
                (preview_w * (1 + len(pose_images) * 2), preview_h),
                "white",
            )
            canvas.paste(ref_image.resize((preview_w, preview_h)), (0, 0))
            for idx, pose_image in enumerate(pose_images):
                pose_preview = pose_image.resize((preview_w, preview_h))
                gen_preview = Image.fromarray(
                    (generated[idx, :, 0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                ).resize((preview_w, preview_h))
                canvas.paste(pose_preview, ((idx * 2 + 1) * preview_w, 0))
                canvas.paste(gen_preview, ((idx * 2 + 2) * preview_w, 0))
            canvas.save(out_dir / ref_path.name)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with=cfg.logging.log_with,
        project_dir=str(output_dir / cfg.logging.project_dir),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    if cfg.seed is not None:
        set_seed(cfg.seed)

    accelerator.init_trackers(
        cfg.logging.run_name,
        config={"config_path": str(Path(args.config).resolve())},
    )

    weight_dtype = get_weight_dtype(cfg.weight_dtype)
    train_scheduler = build_scheduler(cfg)

    train_dataset = IPGDataset(
        dataset_specs=resolve_train_sets(cfg),
        image_height=cfg.data.train_width,
        image_width=cfg.data.train_height,
        reid_height=cfg.data.reid_height,
        reid_width=cfg.data.reid_width,
        image_extensions=cfg.data.get("image_extensions"),
        ref_random_flip=cfg.data.ref_random_flip,
        ref_random_erasing_prob=cfg.data.ref_random_erasing_prob,
        ref_random_erasing_on=cfg.data.ref_random_erasing_on,
        allow_same_reference=cfg.data.allow_same_reference,
        ref_upper_mask_root=cfg_select(
            cfg, "features.part_reference_bank.masks.ref_upper_root"
        ),
        ref_lower_mask_root=cfg_select(
            cfg, "features.part_reference_bank.masks.ref_lower_root"
        ),
        target_upper_mask_root=cfg_select(
            cfg, "features.part_reference_bank.masks.target_upper_root"
        ),
        target_lower_mask_root=cfg_select(
            cfg, "features.part_reference_bank.masks.target_lower_root"
        ),
        color_json_path=cfg_select(cfg, "features.color_structure.color_json_path"),
    )
    if accelerator.is_main_process:
        for summary in train_dataset.dataset_summaries:
            logger.info(
                "Dataset %s: identities=%s, targets=%s, dropped_missing_ref=%s, dropped_missing_pose=%s",
                summary["name"],
                summary["num_identities"],
                summary["num_targets"],
                summary["dropped_missing_ref"],
                summary["dropped_missing_pose"],
            )
        logger.info("Total training samples: %s", len(train_dataset))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.solver.train_batch_size,
        shuffle=True,
        num_workers=cfg.solver.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        accelerator.device, dtype=weight_dtype
    )
    vae.eval()
    vae.requires_grad_(False)

    reid_model = load_reid_model(cfg, accelerator.device)
    part_bank_enabled = bool(
        cfg_select(cfg, "features.part_reference_bank.enabled", False)
    )
    color_structure_enabled = bool(
        cfg_select(cfg, "features.color_structure.enabled", False)
    )
    pose_loss_enabled = bool(cfg_select(cfg, "features.pose_loss.enabled", False))
    color_encoder = None
    if color_structure_enabled:
        color_encoder = ColorTextEncoder(
            cfg_select(cfg, "features.color_structure.clip_model_path", cfg.base_model_path),
            tokenizer_subfolder=cfg_select(
                cfg, "features.color_structure.tokenizer_subfolder", "tokenizer"
            ),
            text_encoder_subfolder=cfg_select(
                cfg, "features.color_structure.text_encoder_subfolder", "text_encoder"
            ),
        ).to(device=accelerator.device, dtype=weight_dtype)
        color_encoder.eval()
    pose_loss_fn = None
    if pose_loss_enabled:
        pose_loss_fn = DifferentiablePoseLoss(
            weight=cfg_select(cfg, "features.pose_loss.weight", 0.05),
            max_sigma=cfg_select(cfg, "features.pose_loss.max_sigma", 0.35),
            distance_iterations=cfg_select(
                cfg, "features.pose_loss.distance_iterations", 8
            ),
        ).to(accelerator.device)

    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    )
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
        },
    )
    pose_guider = PoseGuider(conditioning_embedding_channels=320)
    ifr = IFR(
        input_dim=cfg.model.ifr_input_dim,
        num_tokens=cfg.model.ifr_num_tokens,
        hidden_dim=cfg.model.ifr_hidden_dim,
    )

    if cfg.model.get("reference_unet_ckpt"):
        reference_unet.load_state_dict(
            torch.load(cfg.model.reference_unet_ckpt, map_location="cpu"), strict=True
        )
    if cfg.model.get("denoising_unet_ckpt"):
        denoising_unet.load_state_dict(
            torch.load(cfg.model.denoising_unet_ckpt, map_location="cpu"), strict=True
        )
    if cfg.model.get("pose_guider_ckpt"):
        pose_guider.load_state_dict(
            torch.load(cfg.model.pose_guider_ckpt, map_location="cpu"), strict=True
        )
    if cfg.model.get("ifr_ckpt"):
        ifr.load_state_dict(torch.load(cfg.model.ifr_ckpt, map_location="cpu"), strict=True)

    if cfg.solver.enable_xformers_memory_efficient_attention:
        maybe_enable_xformers(reference_unet, denoising_unet)

    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

    raw_max_sigma = cfg_select(cfg, "features.color_structure.timestep_max_sigma")
    color_timestep_max_sigma = (
        float(raw_max_sigma)
        if raw_max_sigma is not None and float(raw_max_sigma) >= 0
        else None
    )
    ipg_model = IPGTrainModel(
        ifr=ifr,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        part_bank_enabled=part_bank_enabled,
        part_bank_lambda_init=cfg_select(
            cfg, "features.part_reference_bank.lambda_init", 0.0
        ),
        color_structure_enabled=color_structure_enabled,
        color_scale=cfg_select(cfg, "features.color_structure.color_scale", 0.25),
        color_timestep_max_sigma=color_timestep_max_sigma,
        color_min_layer_weight=float(
            cfg_select(cfg, "features.color_structure.min_layer_weight", 0.0)
        ),
        color_hard_gating=bool(
            cfg_select(cfg, "features.color_structure.hard_query_gating", False)
        ),
        color_query_threshold=float(
            cfg_select(cfg, "features.color_structure.query_threshold", 0.5)
        ),
        noise_scheduler=train_scheduler,
    )
    part_bank_fusion_ckpt = cfg_select(cfg, "model.part_bank_fusion_ckpt")
    if part_bank_fusion_ckpt:
        ipg_model.part_bank_fusion.load_state_dict(
            torch.load(part_bank_fusion_ckpt, map_location="cpu"), strict=True
        )

    params_to_optimize = [p for p in ipg_model.parameters() if p.requires_grad]
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError("bitsandbytes is required when use_8bit_adam=True") from exc
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        params_to_optimize,
        lr=cfg.solver.learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=cfg.solver.max_train_steps * accelerator.num_processes,
    )

    ipg_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        ipg_model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = maybe_resume(
        accelerator=accelerator,
        output_dir=output_dir,
        resume_from_checkpoint=cfg.model.resume_from_checkpoint,
    )
    first_epoch = global_step // num_update_steps_per_epoch
    resume_step = (
        global_step % num_update_steps_per_epoch
    ) * cfg.solver.gradient_accumulation_steps

    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="IPG training",
    )

    for epoch in range(
        first_epoch,
        math.ceil(cfg.solver.max_train_steps / num_update_steps_per_epoch) + 1,
    ):
        ipg_model.train()
        for step, batch in enumerate(train_dataloader):
            if epoch == first_epoch and step < resume_step:
                continue

            with accelerator.accumulate(ipg_model):
                target_images = batch["target_image"].to(
                    accelerator.device, dtype=weight_dtype
                )
                ref_images = batch["ref_image"].to(accelerator.device, dtype=weight_dtype)
                pose_images = batch["pose_image"].to(
                    accelerator.device, dtype=weight_dtype
                )
                reid_images = batch["reid_image"].to(
                    accelerator.device, dtype=torch.float32
                )
                ref_upper_masks = batch["ref_upper_mask"].to(
                    accelerator.device, dtype=weight_dtype
                )
                ref_lower_masks = batch["ref_lower_mask"].to(
                    accelerator.device, dtype=weight_dtype
                )
                target_upper_masks = batch["target_upper_mask"].to(
                    accelerator.device, dtype=weight_dtype
                )
                target_lower_masks = batch["target_lower_mask"].to(
                    accelerator.device, dtype=weight_dtype
                )

                with torch.no_grad():
                    latents = vae.encode(target_images).latent_dist.sample()
                    latents = latents * cfg.model.latent_scale
                    ref_latents = vae.encode(ref_images).latent_dist.sample()
                    ref_latents = ref_latents * cfg.model.latent_scale
                    ref_upper_latents = None
                    ref_lower_latents = None
                    if part_bank_enabled:
                        ref_upper_images = ref_images * ref_upper_masks
                        ref_lower_images = ref_images * ref_lower_masks
                        ref_upper_latents = vae.encode(
                            ref_upper_images
                        ).latent_dist.sample()
                        ref_upper_latents = ref_upper_latents * cfg.model.latent_scale
                        ref_lower_latents = vae.encode(
                            ref_lower_images
                        ).latent_dist.sample()
                        ref_lower_latents = ref_lower_latents * cfg.model.latent_scale
                    reid_features = build_reid_condition(reid_model, reid_images)
                    color_upper_states = None
                    color_lower_states = None
                    if color_encoder is not None:
                        color_upper_states, color_lower_states = color_encoder.encode_pair(
                            batch["color_upper_text"],
                            batch["color_lower_text"],
                            device=accelerator.device,
                            dtype=weight_dtype,
                        )

                noise = torch.randn_like(latents)
                if cfg.noise_offset > 0:
                    noise = noise + cfg.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1),
                        device=latents.device,
                        dtype=latents.dtype,
                    )
                timesteps = torch.randint(
                    0,
                    train_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                    dtype=torch.long,
                )
                noisy_latents = train_scheduler.add_noise(latents, noise, timesteps).unsqueeze(2)

                prediction_type = train_scheduler.config.prediction_type
                if prediction_type == "epsilon":
                    target = noise
                elif prediction_type == "v_prediction":
                    target = train_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unsupported prediction type: {prediction_type}")

                model_pred = ipg_model(
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    ref_image_latents=ref_latents,
                    reid_features=reid_features.to(
                        dtype=accelerator.unwrap_model(ipg_model).ifr.proj_motion.weight.dtype
                    ),
                    pose_images=pose_images,
                    ref_upper_latents=ref_upper_latents,
                    ref_lower_latents=ref_lower_latents,
                    target_upper_mask=target_upper_masks,
                    target_lower_mask=target_lower_masks,
                    color_upper_states=color_upper_states,
                    color_lower_states=color_lower_states,
                    cond_dropout_prob=cfg.uncond_ratio,
                )

                target = target.unsqueeze(2)
                if cfg.snr_gamma is None:
                    loss_mse = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    snr = compute_snr(train_scheduler, timesteps)
                    if prediction_type == "epsilon":
                        mse_loss_weights = torch.minimum(
                            snr, torch.full_like(snr, cfg.snr_gamma)
                        ) / snr
                    else:
                        mse_loss_weights = torch.minimum(
                            snr, torch.full_like(snr, cfg.snr_gamma)
                        ) / (snr + 1)

                    loss_mse = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss_mse = loss_mse.mean(dim=list(range(1, loss_mse.ndim)))
                    loss_mse = (loss_mse * mse_loss_weights).mean()

                loss_pose = model_pred.new_zeros(())
                if pose_loss_fn is not None:
                    loss_pose = pose_loss_fn(
                        model_pred=model_pred,
                        noisy_latents=noisy_latents,
                        timesteps=timesteps,
                        scheduler=train_scheduler,
                        vae=vae,
                        pose_images=pose_images,
                        prediction_type=prediction_type,
                        latent_scale=cfg.model.latent_scale,
                    )
                loss = loss_mse + (
                    pose_loss_fn.weight * loss_pose if pose_loss_fn is not None else 0
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        ipg_model.parameters(), cfg.solver.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                logs = {
                    "loss": loss.detach().item(),
                    "loss_mse": loss_mse.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                }
                if pose_loss_fn is not None:
                    logs["loss_pose"] = loss_pose.detach().item()
                progress_bar.set_postfix(loss=f"{logs['loss']:.4f}", lr=f"{logs['lr']:.2e}")
                accelerator.log(logs, step=global_step)

                if (
                    global_step % cfg.solver.checkpointing_steps == 0
                    or global_step == cfg.solver.max_train_steps
                ):
                    checkpoint_dir = save_component_weights(
                        output_dir=output_dir,
                        global_step=global_step,
                        accelerator=accelerator,
                        ipg_model=ipg_model,
                    )
                    if accelerator.is_main_process:
                        delete_additional_ckpt(
                            str(output_dir), cfg.solver.checkpoints_total_limit
                        )
                        logger.info(f"Saved checkpoint to {checkpoint_dir}")

                if (
                    cfg.validation.enabled
                    and global_step % cfg.validation.every_n_steps == 0
                ):
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        run_validation(
                            cfg=cfg,
                            accelerator=accelerator,
                            vae=vae,
                            reid_model=reid_model,
                            ipg_model=ipg_model,
                            output_dir=output_dir,
                            global_step=global_step,
                        )
                    accelerator.wait_for_everyone()

            if global_step >= cfg.solver.max_train_steps:
                break

        if global_step >= cfg.solver.max_train_steps:
            break

    accelerator.wait_for_everyone()
    save_component_weights(
        output_dir=output_dir,
        global_step=global_step,
        accelerator=accelerator,
        ipg_model=ipg_model,
    )
    accelerator.end_training()


if __name__ == "__main__":
    main()
