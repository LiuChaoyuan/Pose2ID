"""Generate Market1501 tiny images and record text/body-part cross-attention."""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF

import Market_gen as market
from src.utils.mask_utils import mask_to_token_weights, pose_to_part_masks, prepare_mask_batch


class CrossAttentionRecorder:
    """Records color-prompt cross-attention at every denoising step."""

    def __init__(
        self,
        output_dir: Path,
        *,
        split_name: str,
        max_attention_layers: int = 6,
        mask_threshold: float = 0.5,
    ):
        self.output_dir = Path(output_dir)
        self.split_name = split_name
        self.max_attention_layers = max(1, int(max_attention_layers))
        self.mask_threshold = float(mask_threshold)
        self.rows: List[dict] = []
        self.token_rows: List[dict] = []
        self.hooks = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reset_case()

    def reset_case(self) -> None:
        self.case_name = ""
        self.pose_names: List[str] = []
        self.upper_prompts: List[dict] = []
        self.lower_prompts: List[dict] = []
        self.target_upper_mask = None
        self.target_lower_mask = None
        self.batch_size = 0
        self.current_step = -1
        self.current_timestep = None
        self.prompt_call_index = 0
        self.step_layers = set()

    def attach(self, denoising_unet: torch.nn.Module) -> None:
        self.hooks.append(
            denoising_unet.register_forward_pre_hook(
                self._denoising_pre_hook,
                with_kwargs=True,
            )
        )
        for name, module in denoising_unet.named_modules():
            if name.endswith("attn2") and all(hasattr(module, attr) for attr in ("to_q", "to_k")):
                self.hooks.append(
                    module.register_forward_hook(
                        self._make_attention_hook(name),
                        with_kwargs=True,
                    )
                )

    def close(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def begin_case(
        self,
        *,
        image_name: str,
        pose_names: Sequence[str],
        upper_texts: Sequence[str],
        lower_texts: Sequence[str],
        tokenizer,
        target_upper_mask: torch.Tensor,
        target_lower_mask: torch.Tensor,
    ) -> None:
        self.reset_case()
        self.case_name = image_name
        self.pose_names = list(pose_names)
        self.upper_prompts = self._encode_prompt_records(upper_texts, tokenizer)
        self.lower_prompts = self._encode_prompt_records(lower_texts, tokenizer)
        self.target_upper_mask = target_upper_mask.detach()
        self.target_lower_mask = target_lower_mask.detach()
        self.batch_size = len(self.pose_names)
        self._save_mask_previews()

    def _encode_prompt_records(self, texts: Sequence[str], tokenizer) -> List[dict]:
        if tokenizer is None:
            return [{"text": str(text or ""), "tokens": [], "valid_mask": []} for text in texts]
        encoded = tokenizer(
            [str(text or "") for text in texts],
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        special_ids = set(getattr(tokenizer, "all_special_ids", []))
        records = []
        for text, ids, attn_mask in zip(
            texts,
            encoded["input_ids"].cpu(),
            encoded["attention_mask"].cpu().bool(),
        ):
            tokens = tokenizer.convert_ids_to_tokens(ids.tolist())
            valid = [
                bool(mask.item()) and int(token_id.item()) not in special_ids
                for token_id, mask in zip(ids, attn_mask)
            ]
            if not any(valid):
                valid = attn_mask.tolist()
            records.append({"text": str(text or ""), "tokens": tokens, "valid_mask": valid})
        return records

    def _denoising_pre_hook(self, _module, args, kwargs) -> None:
        self.current_step += 1
        self.current_timestep = None
        self.prompt_call_index = 0
        self.step_layers = set()
        timestep = args[1] if len(args) > 1 else kwargs.get("timestep")
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.detach().flatten()[0].item()
        if timestep is not None:
            self.current_timestep = int(timestep)

    def _make_attention_hook(self, layer_name: str):
        def hook(module, args, kwargs, _output):
            self._capture_attention(layer_name, module, args, kwargs)

        return hook

    def _capture_attention(self, layer_name: str, module, args, kwargs) -> None:
        if self.batch_size <= 0:
            return
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        encoder_states = kwargs.get("encoder_hidden_states")
        attention_mask = kwargs.get("attention_mask")
        if encoder_states is None and len(args) > 1:
            encoder_states = args[1]
        if hidden_states is None or encoder_states is None or encoder_states.ndim != 3:
            return

        prompt_part = self._detect_prompt_part(encoder_states)
        if prompt_part is None:
            return
        if layer_name not in self.step_layers:
            if len(self.step_layers) >= self.max_attention_layers:
                return
            self.step_layers.add(layer_name)

        with torch.no_grad():
            attention_probs = self._compute_attention_probs(
                module,
                hidden_states.detach(),
                encoder_states.detach(),
                attention_mask.detach() if isinstance(attention_mask, torch.Tensor) else attention_mask,
            )
        if attention_probs is not None:
            self._store_attention(layer_name, prompt_part, hidden_states, attention_probs)

    def _detect_prompt_part(self, encoder_states: torch.Tensor) -> Optional[str]:
        # ReID identity tokens are short. Color prompts are CLIP token sequences
        # and are called as upper/lower in that order inside mutual_self_attention.
        token_count = encoder_states.shape[1]
        expected = {
            len(records[0]["tokens"])
            for records in (self.upper_prompts, self.lower_prompts)
            if records and records[0].get("tokens")
        }
        if expected and token_count not in expected:
            return None
        if not expected and token_count <= 32:
            return None

        prompt_part = "upper" if self.prompt_call_index % 2 == 0 else "lower"
        self.prompt_call_index += 1
        return prompt_part

    def _compute_attention_probs(
        self,
        module,
        hidden_states: torch.Tensor,
        encoder_states: torch.Tensor,
        attention_mask,
    ) -> Optional[torch.Tensor]:
        if hidden_states.ndim == 4:
            batch, channels, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch, channels, height * width).transpose(1, 2)
        if hidden_states.ndim != 3:
            return None

        try:
            if getattr(module, "norm_cross", False) and hasattr(module, "norm_encoder_hidden_states"):
                encoder_states = module.norm_encoder_hidden_states(encoder_states)
            query = module.to_q(hidden_states)
            key = module.to_k(encoder_states)
        except Exception:
            return None

        heads = int(getattr(module, "heads", 1))
        query = self._head_to_batch_dim(module, query, heads).float()
        key = self._head_to_batch_dim(module, key, heads).float()
        attention_mask = self._prepare_attention_mask(
            module, attention_mask, key.shape[1], hidden_states.shape[0]
        )

        if hasattr(module, "get_attention_scores"):
            try:
                return module.get_attention_scores(query, key, attention_mask).detach()
            except Exception:
                pass

        scale = float(getattr(module, "scale", query.shape[-1] ** -0.5))
        scores = torch.baddbmm(
            torch.empty(
                query.shape[0],
                query.shape[1],
                key.shape[1],
                dtype=query.dtype,
                device=query.device,
            ),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=scale,
        )
        if attention_mask is not None:
            scores = scores + attention_mask.float()
        return scores.softmax(dim=-1).detach()

    def _store_attention(
        self,
        layer_name: str,
        prompt_part: str,
        hidden_states: torch.Tensor,
        attention_probs: torch.Tensor,
    ) -> None:
        if hidden_states.ndim == 4:
            hidden_batch = hidden_states.shape[0]
            query_tokens = hidden_states.shape[-2] * hidden_states.shape[-1]
        else:
            hidden_batch = hidden_states.shape[0]
            query_tokens = hidden_states.shape[1]
        if attention_probs.shape[0] % hidden_batch != 0:
            return

        heads = attention_probs.shape[0] // hidden_batch
        probs = attention_probs.view(hidden_batch, heads, query_tokens, -1).mean(dim=1)
        if hidden_batch == self.batch_size * 2:
            probs = probs[self.batch_size :]
        elif hidden_batch != self.batch_size:
            return

        prompt_records = self.upper_prompts if prompt_part == "upper" else self.lower_prompts
        for region_name, weights in self._region_weights(query_tokens, probs.device).items():
            weights = weights.to(device=probs.device, dtype=probs.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            token_scores = torch.einsum("bq,bqk->bk", weights, probs).float().cpu()

            for sample_idx in range(self.batch_size):
                prompt = prompt_records[sample_idx] if sample_idx < len(prompt_records) else {}
                valid_indices = self._valid_token_indices(prompt, token_scores.shape[1])
                sample_scores = token_scores[sample_idx]
                valid_scores = sample_scores[valid_indices]
                row = {
                    "split": self.split_name,
                    "image": self.case_name,
                    "pose": self.pose_names[sample_idx],
                    "sample": sample_idx,
                    "step": self.current_step,
                    "timestep": self.current_timestep,
                    "layer": layer_name,
                    "prompt_part": prompt_part,
                    "region": region_name,
                    "prompt_text": prompt.get("text", ""),
                    "valid_token_mean": float(valid_scores.mean().item()),
                    "valid_token_sum": float(valid_scores.sum().item()),
                    "max_token_score": float(valid_scores.max().item()),
                    "min_token_score": float(valid_scores.min().item()),
                    "query_tokens": int(query_tokens),
                    "key_tokens": int(token_scores.shape[1]),
                    "heads": int(heads),
                }
                self.rows.append(row)
                for token_index in valid_indices:
                    self.token_rows.append(
                        {
                            "split": self.split_name,
                            "image": self.case_name,
                            "pose": self.pose_names[sample_idx],
                            "sample": sample_idx,
                            "step": self.current_step,
                            "timestep": self.current_timestep,
                            "layer": layer_name,
                            "prompt_part": prompt_part,
                            "region": region_name,
                            "token_index": int(token_index),
                            "token": self._token_at(prompt, token_index),
                            "score": float(sample_scores[token_index].item()),
                        }
                    )

    def _region_weights(self, query_tokens: int, device: torch.device) -> Dict[str, torch.Tensor]:
        weights = {"all": torch.ones(self.batch_size, query_tokens, device=device)}
        for region_name, mask in (
            ("target_upper", self.target_upper_mask),
            ("target_lower", self.target_lower_mask),
        ):
            token_weights = mask_to_token_weights(
                mask,
                query_tokens,
                hidden_batch=self.batch_size,
                dtype=torch.float32,
                device=device,
            )
            if token_weights is not None:
                weights[region_name] = token_weights.squeeze(-1)
        return weights

    def _valid_token_indices(self, prompt: dict, key_tokens: int) -> List[int]:
        valid_mask = list(prompt.get("valid_mask") or [True] * key_tokens)
        if len(valid_mask) < key_tokens:
            valid_mask.extend([False] * (key_tokens - len(valid_mask)))
        indices = [idx for idx, valid in enumerate(valid_mask[:key_tokens]) if valid]
        return indices or list(range(key_tokens))

    def _save_mask_previews(self) -> None:
        preview_dir = self.output_dir / "mask_previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        for mask_name, mask in (
            ("target_upper", self.target_upper_mask),
            ("target_lower", self.target_lower_mask),
        ):
            if mask is None:
                continue
            tensor = mask.detach().float().clamp(0, 1).cpu()
            for idx in range(min(tensor.shape[0], len(self.pose_names))):
                image = Image.fromarray(
                    (tensor[idx, 0].numpy() * 255).astype(np.uint8),
                    mode="L",
                ).convert("RGB")
                active = tensor[idx, 0] > self.mask_threshold
                ys, xs = torch.where(active)
                if xs.numel() > 0:
                    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                    ImageDraw.Draw(image).rectangle(bbox, outline=(255, 0, 0), width=2)
                image.save(
                    preview_dir
                    / f"{_safe_name(self.case_name)}_{_safe_name(self.pose_names[idx])}_{mask_name}.png"
                )

    def save_outputs(self) -> None:
        _write_csv(
            self.output_dir / "attention_region_scores.csv",
            self.rows,
            [
                "split",
                "image",
                "pose",
                "sample",
                "step",
                "timestep",
                "layer",
                "prompt_part",
                "region",
                "prompt_text",
                "valid_token_mean",
                "valid_token_sum",
                "max_token_score",
                "min_token_score",
                "query_tokens",
                "key_tokens",
                "heads",
            ],
        )
        _write_csv(
            self.output_dir / "attention_token_scores.csv",
            self.token_rows,
            [
                "split",
                "image",
                "pose",
                "sample",
                "step",
                "timestep",
                "layer",
                "prompt_part",
                "region",
                "token_index",
                "token",
                "score",
            ],
        )
        with open(self.output_dir / "attention_region_scores.jsonl", "w", encoding="utf-8") as handle:
            for row in self.rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        plot_attention_lines(self.rows, self.output_dir / "attention_by_step.png")
        plot_attention_heatmaps(self.rows, self.output_dir)
        plot_final_token_bars(self.token_rows, self.output_dir)

    def _prepare_attention_mask(self, module, attention_mask, target_length: int, batch_size: int):
        if attention_mask is None or not hasattr(module, "prepare_attention_mask"):
            return attention_mask
        try:
            return module.prepare_attention_mask(attention_mask, target_length, batch_size)
        except TypeError:
            return module.prepare_attention_mask(attention_mask, target_length)
        except Exception:
            return attention_mask

    def _head_to_batch_dim(self, module, tensor: torch.Tensor, heads: int) -> torch.Tensor:
        if hasattr(module, "head_to_batch_dim"):
            try:
                return module.head_to_batch_dim(tensor)
            except TypeError:
                return module.head_to_batch_dim(tensor, out_dim=3)
        batch_size, tokens, channels = tensor.shape
        head_dim = channels // heads
        tensor = tensor.view(batch_size, tokens, heads, head_dim)
        return tensor.permute(0, 2, 1, 3).reshape(batch_size * heads, tokens, head_dim)

    def _token_at(self, prompt: dict, token_index: int) -> str:
        tokens = prompt.get("tokens") or []
        return tokens[token_index] if token_index < len(tokens) else str(token_index)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Market1501 tiny query/gallery images with cross-attention details."
    )
    parser.add_argument("--tiny_root", type=str, default="/root/autodl-tmp/datasets/market1501_tiny")
    parser.add_argument("--query_dir", type=str, default=None)
    parser.add_argument("--bound_box_test_dir", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default="/root/autodl-fs/ipg_trained_less_05-04/checkpoint-3000")
    parser.add_argument("--pose_dir", type=str, default="standard_poses")
    parser.add_argument("--config", type=str, default="./configs/inference.yaml")
    parser.add_argument("--reid_cfg_path", type=str, default="./cfg_transreid.pkl")
    parser.add_argument("--reid_ckpt_name", type=str, default="transformer_20.pth")
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--color_json", type=str, default=None)
    parser.add_argument("--mask_root", type=str, default=None)
    parser.add_argument("--detail_root", type=str, default=None)
    parser.add_argument("--max_attention_layers", type=int, default=6)
    parser.add_argument("--max_images_per_split", type=int, default=None)
    parser.add_argument("--disable_part_bank", action="store_true")
    parser.add_argument("--disable_color_structure", action="store_true")
    return parser.parse_args()


def resolve_tiny_paths(args) -> None:
    tiny_root = Path(args.tiny_root)
    if args.query_dir is None:
        args.query_dir = str(tiny_root / "query_tiny")
    if args.bound_box_test_dir is None:
        args.bound_box_test_dir = str(tiny_root / "bounding_box_test_tiny")
    if args.color_json is None:
        args.color_json = str(tiny_root / "clothing_colors_tiny_nl.json")
    if args.mask_root is None:
        args.mask_root = str(tiny_root / "sam3_part_masks")
    if args.detail_root is None:
        args.detail_root = str(tiny_root / "attention_details_tiny")


def prepare_target_masks(
    mask_root: Optional[str],
    pose_items: Sequence[Tuple[str, Image.Image, Path]],
    *,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, List[Image.Image], List[Image.Image]]:
    upper_images = []
    lower_images = []
    if mask_root:
        for _, _, pose_path in pose_items:
            upper, lower = market.load_mask_pair(mask_root, pose_path, group_name="standard_poses")
            if upper is None or lower is None:
                upper_images = []
                lower_images = []
                break
            upper_images.append(upper)
            lower_images.append(lower)

    if upper_images and lower_images:
        upper_tensor = prepare_mask_batch(
            upper_images,
            len(pose_items),
            height,
            width,
            device=device,
            dtype=dtype,
            fallback="upper",
        )
        lower_tensor = prepare_mask_batch(
            lower_images,
            len(pose_items),
            height,
            width,
            device=device,
            dtype=dtype,
            fallback="lower",
        )
        return upper_tensor, lower_tensor, upper_images, lower_images

    pose_tensors = []
    for _, pose_image, _ in pose_items:
        pose_tensors.append(TF.to_tensor(TF.resize(pose_image, [height, width])))
    pose_batch = torch.stack(pose_tensors).to(device=device, dtype=dtype)
    upper_tensor, lower_tensor = pose_to_part_masks(pose_batch)
    return upper_tensor, lower_tensor, [], []


def save_preview(
    ref_image: Image.Image,
    pose_items: Sequence[Tuple[str, Image.Image, Path]],
    generated_images: torch.Tensor,
    out_path: Path,
) -> None:
    tile_w, tile_h = 128, 256
    canvas = Image.new("RGB", (tile_w * (1 + len(pose_items) * 2), tile_h), "white")
    canvas.paste(ref_image.resize((tile_w, tile_h)), (0, 0))
    for idx, (_, pose_image, _) in enumerate(pose_items):
        generated = generated_images[idx, :, 0].cpu()
        generated_image = TF.to_pil_image(generated).resize((tile_w, tile_h))
        canvas.paste(pose_image.resize((tile_w, tile_h)), ((idx * 2 + 1) * tile_w, 0))
        canvas.paste(generated_image, ((idx * 2 + 2) * tile_w, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def generate_split_with_attention(
    *,
    split_dir: Path,
    pipe,
    ifr,
    reid_net,
    color_encoder,
    color_descriptions: dict,
    reid_transform,
    pose_items: Sequence[Tuple[str, Image.Image, Path]],
    cfg,
    args,
    generator: torch.Generator,
    device: torch.device,
    part_bank_enabled: bool,
    color_enabled: bool,
):
    if not split_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {split_dir}")

    split_name = split_dir.name
    output_root = split_dir.parent / f"{split_name}_gen"
    preview_root = output_root / "_previews"
    pose_names = [pose_name for pose_name, _, _ in pose_items]
    pose_images = [pose_image.copy() for _, pose_image, _ in pose_items]
    pose_paths = [pose_path for _, _, pose_path in pose_items]
    market.ensure_output_dirs(output_root, pose_names)
    preview_root.mkdir(parents=True, exist_ok=True)

    recorder = CrossAttentionRecorder(
        Path(args.detail_root) / split_name,
        split_name=split_name,
        max_attention_layers=args.max_attention_layers,
    )
    recorder.attach(pipe.denoising_unet)

    image_paths = market.collect_ref_images(split_dir)
    if args.max_images_per_split is not None:
        image_paths = image_paths[: args.max_images_per_split]
    print(f"Processing {split_dir} -> {output_root}, valid images: {len(image_paths)}")

    try:
        for image_index, image_path in enumerate(image_paths, start=1):
            with Image.open(image_path) as handle:
                ref_image = handle.convert("RGB")
            reid_input = reid_transform(ref_image).to(device=device, dtype=torch.float32)
            feature_embeds = market.build_identity_embeddings(
                reid_net=reid_net,
                ifr=ifr,
                reid_tensor=reid_input,
                num_poses=len(pose_items),
                device=device,
            )

            upper_texts = [""] * len(pose_items)
            lower_texts = [""] * len(pose_items)
            color_upper_states = None
            color_lower_states = None
            if color_enabled and color_encoder is not None:
                upper_texts, lower_texts = market.get_color_texts(
                    color_descriptions, image_path, len(pose_items)
                )
                color_upper_states, color_lower_states = color_encoder.encode_pair(
                    upper_texts,
                    lower_texts,
                    device=device,
                    dtype=feature_embeds.dtype,
                )

            ref_upper_mask, ref_lower_mask = market.load_mask_pair(
                args.mask_root, image_path, group_name=split_name
            )
            target_upper_masks = []
            target_lower_masks = []
            if args.mask_root:
                for pose_path in pose_paths:
                    upper_mask, lower_mask = market.load_mask_pair(
                        args.mask_root, pose_path, group_name="standard_poses"
                    )
                    if upper_mask is None or lower_mask is None:
                        target_upper_masks = []
                        target_lower_masks = []
                        break
                    target_upper_masks.append(upper_mask)
                    target_lower_masks.append(lower_mask)

            target_upper_tensor, target_lower_tensor, _, _ = prepare_target_masks(
                args.mask_root,
                pose_items,
                height=cfg.data.train_width,
                width=cfg.data.train_height,
                device=device,
                dtype=feature_embeds.dtype,
            )
            recorder.begin_case(
                image_name=image_path.stem,
                pose_names=pose_names,
                upper_texts=upper_texts,
                lower_texts=lower_texts,
                tokenizer=getattr(color_encoder, "tokenizer", None),
                target_upper_mask=target_upper_tensor,
                target_lower_mask=target_lower_tensor,
            )

            generated_images = pipe(
                feature_embeds,
                [ref_image.copy() for _ in pose_items],
                pose_images,
                cfg.data.train_height,
                cfg.data.train_width,
                args.num_inference_steps,
                args.guidance_scale,
                batch_size=len(pose_items),
                generator=generator,
                ref_upper_mask=[ref_upper_mask.copy() for _ in pose_items]
                if ref_upper_mask is not None
                else None,
                ref_lower_mask=[ref_lower_mask.copy() for _ in pose_items]
                if ref_lower_mask is not None
                else None,
                target_upper_mask=target_upper_masks if target_upper_masks else None,
                target_lower_mask=target_lower_masks if target_lower_masks else None,
                color_upper_states=color_upper_states,
                color_lower_states=color_lower_states,
                part_bank_enabled=part_bank_enabled,
                color_structure_enabled=color_enabled,
                color_scale=market.cfg_select(cfg, "features.color_structure.color_scale", 0.25),
            ).images

            market.save_generated_images(generated_images, pose_names, output_root, image_path.name)
            save_preview(ref_image, pose_items, generated_images, preview_root / f"{image_path.stem}.jpg")
            print(f"[{image_index}/{len(image_paths)}] Saved {image_path.name}")
    finally:
        recorder.close()
        recorder.save_outputs()
        print(f"Attention details saved to {recorder.output_dir}")


def plot_attention_lines(rows: List[dict], out_path: Path) -> None:
    if not rows:
        return
    series = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = f"{row['prompt_part']}->{row['region']}"
        series[key][int(row["step"])].append(float(row["valid_token_sum"]))
    plt.figure(figsize=(10, 5))
    for key, values_by_step in sorted(series.items()):
        steps = sorted(values_by_step)
        values = [sum(values_by_step[step]) / len(values_by_step[step]) for step in steps]
        plt.plot(steps, values, marker="o", linewidth=1.4, label=key)
    plt.xlabel("Denoising step")
    plt.ylabel("Attention mass on prompt tokens")
    plt.title("Text prompt cross-attention by body region")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_attention_heatmaps(rows: List[dict], out_dir: Path) -> None:
    if not rows:
        return
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["image"], row["pose"])].append(row)
    for (image_name, pose_name), items in grouped.items():
        labels = sorted({f"{row['prompt_part']}->{row['region']}" for row in items})
        steps = sorted({int(row["step"]) for row in items})
        matrix = np.zeros((len(steps), len(labels)), dtype=np.float32)
        counts = np.zeros_like(matrix)
        label_index = {label: idx for idx, label in enumerate(labels)}
        step_index = {step: idx for idx, step in enumerate(steps)}
        for row in items:
            y = step_index[int(row["step"])]
            x = label_index[f"{row['prompt_part']}->{row['region']}"]
            matrix[y, x] += float(row["valid_token_sum"])
            counts[y, x] += 1
        matrix = matrix / np.maximum(counts, 1)

        plt.figure(figsize=(max(8, len(labels) * 1.1), max(4, len(steps) * 0.32)))
        plt.imshow(matrix, aspect="auto", cmap="viridis")
        plt.colorbar(label="attention mass")
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
        plt.yticks(range(len(steps)), steps)
        plt.xlabel("Prompt and image region")
        plt.ylabel("Denoising step")
        plt.title(f"{image_name} / {pose_name}")
        plt.tight_layout()
        plt.savefig(
            out_dir / f"attention_heatmap_{_safe_name(image_name)}_{_safe_name(pose_name)}.png",
            dpi=160,
        )
        plt.close()


def plot_final_token_bars(token_rows: List[dict], out_dir: Path) -> None:
    if not token_rows:
        return
    grouped = defaultdict(list)
    for row in token_rows:
        grouped[(row["image"], row["pose"], row["prompt_part"], row["region"])].append(row)
    made = 0
    for key, items in sorted(grouped.items()):
        image_name, pose_name, prompt_part, region = key
        if region == "all":
            continue
        latest_step = max(int(row["step"]) for row in items)
        latest = [row for row in items if int(row["step"]) == latest_step]
        token_scores = defaultdict(list)
        for row in latest:
            token_scores[row["token"]].append(float(row["score"]))
        averaged = sorted(
            ((token, sum(scores) / len(scores)) for token, scores in token_scores.items()),
            key=lambda item: item[1],
            reverse=True,
        )[:20]
        if not averaged:
            continue
        labels, values = zip(*averaged)
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(values)), values)
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
        plt.ylabel("Average attention")
        plt.title(f"{image_name} / {pose_name}: {prompt_part} prompt on {region}, step {latest_step}")
        plt.tight_layout()
        plt.savefig(
            out_dir
            / f"token_bar_{_safe_name(image_name)}_{_safe_name(pose_name)}_{prompt_part}_{region}.png",
            dpi=160,
        )
        plt.close()
        made += 1
        if made >= 24:
            break


def _write_csv(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


def main():
    args = parse_args()
    resolve_tiny_paths(args)
    cfg = market.load_config(args.config)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Market_gen_tiny.py.")

    device = torch.device("cuda")
    if cfg.seed is not None:
        market.seed_everything(cfg.seed)

    part_bank_enabled = bool(
        market.cfg_select(cfg, "features.part_reference_bank.enabled", False)
    ) and not args.disable_part_bank
    color_enabled = bool(
        market.cfg_select(cfg, "features.color_structure.enabled", False)
    ) and not args.disable_color_structure
    color_descriptions = (
        market.load_color_descriptions(args.color_json) if color_enabled else {}
    )

    pipe, ifr, reid_net, color_encoder = market.load_models(
        cfg, args, device, part_bank_enabled, color_enabled
    )
    reid_transform = market.build_reid_transform()
    pose_items = market.collect_pose_images(Path(args.pose_dir))
    generator = torch.Generator(device=device)
    if cfg.seed is not None:
        generator.manual_seed(cfg.seed)

    with torch.inference_mode():
        generate_split_with_attention(
            split_dir=Path(args.query_dir),
            pipe=pipe,
            ifr=ifr,
            reid_net=reid_net,
            color_encoder=color_encoder,
            color_descriptions=color_descriptions,
            reid_transform=reid_transform,
            pose_items=pose_items,
            cfg=cfg,
            args=args,
            generator=generator,
            device=device,
            part_bank_enabled=part_bank_enabled,
            color_enabled=color_enabled,
        )
        generate_split_with_attention(
            split_dir=Path(args.bound_box_test_dir),
            pipe=pipe,
            ifr=ifr,
            reid_net=reid_net,
            color_encoder=color_encoder,
            color_descriptions=color_descriptions,
            reid_transform=reid_transform,
            pose_items=pose_items,
            cfg=cfg,
            args=args,
            generator=generator,
            device=device,
            part_bank_enabled=part_bank_enabled,
            color_enabled=color_enabled,
        )


if __name__ == "__main__":
    main()
