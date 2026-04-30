# Adapted from https://github.com/magic-research/magic-animate/blob/main/magicanimate/models/mutual_self_attention.py
from typing import Any, Dict, Optional

import torch
from einops import rearrange

from src.models.attention import TemporalBasicTransformerBlock
from src.utils.mask_utils import mask_to_token_weights

from .attention import BasicTransformerBlock


def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


class PartBankFusion(torch.nn.Module):
    def __init__(self, lambda_init: float = 0.0):
        super().__init__()
        self.lambda_upper = torch.nn.Parameter(torch.tensor(float(lambda_init)))
        self.lambda_lower = torch.nn.Parameter(torch.tensor(float(lambda_init)))

    def forward(
        self,
        global_features: torch.Tensor,
        upper_features: Optional[torch.Tensor],
        lower_features: Optional[torch.Tensor],
        upper_mask: Optional[torch.Tensor],
        lower_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        fused = global_features
        if upper_features is not None and upper_mask is not None:
            fused = fused + self.lambda_upper * upper_mask * (upper_features - global_features)
        if lower_features is not None and lower_mask is not None:
            fused = fused + self.lambda_lower * lower_mask * (lower_features - global_features)
        return fused


class ReferenceAttentionControl:
    def __init__(
        self,
        unet,
        mode="write",
        do_classifier_free_guidance=False,
        attention_auto_machine_weight=float("inf"),
        gn_auto_machine_weight=1.0,
        style_fidelity=1.0,
        reference_attn=True,
        reference_adain=False,
        fusion_blocks="midup",
        batch_size=1,
        part_bank_enabled=False,
        part_bank_fusion: Optional[PartBankFusion] = None,
        color_structure_enabled=False,
        color_scale=0.0,
    ) -> None:
        # 10. Modify self attention and group norm
        self.unet = unet
        assert mode in ["read", "write"]
        assert fusion_blocks in ["midup", "full"]
        self.reference_attn = reference_attn
        self.reference_adain = reference_adain
        self.fusion_blocks = fusion_blocks
        self.part_bank_enabled = part_bank_enabled
        self.part_bank_fusion = part_bank_fusion
        self.color_structure_enabled = color_structure_enabled
        self.color_scale = float(color_scale)
        self.target_upper_mask = None
        self.target_lower_mask = None
        self.color_upper_states = None
        self.color_lower_states = None
        self.register_reference_hooks(
            mode,
            do_classifier_free_guidance,
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            style_fidelity,
            reference_attn,
            reference_adain,
            fusion_blocks,
            batch_size=batch_size,
        )

    def register_reference_hooks(
        self,
        mode,
        do_classifier_free_guidance,
        attention_auto_machine_weight,
        gn_auto_machine_weight,
        style_fidelity,
        reference_attn,
        reference_adain,
        dtype=torch.float16,
        batch_size=1,
        num_images_per_prompt=1,
        device=torch.device("cpu"),
        fusion_blocks="midup",
    ):
        MODE = mode
        do_classifier_free_guidance = do_classifier_free_guidance
        attention_auto_machine_weight = attention_auto_machine_weight
        gn_auto_machine_weight = gn_auto_machine_weight
        style_fidelity = style_fidelity
        reference_attn = reference_attn
        reference_adain = reference_adain
        fusion_blocks = fusion_blocks
        num_images_per_prompt = num_images_per_prompt
        dtype = dtype
        controller = self
        if do_classifier_free_guidance:
            uc_mask = (
                torch.Tensor(
                    [1] * batch_size * num_images_per_prompt * 16
                    + [0] * batch_size * num_images_per_prompt * 16
                )
                .to(device)
                .bool()
            )
        else:
            uc_mask = (
                torch.Tensor([0] * batch_size * num_images_per_prompt * 2)
                .to(device)
                .bool()
            )

        def hacked_basic_transformer_inner_forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            video_length=None,
        ):
            if self.use_ada_layer_norm:  # False
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                (
                    norm_hidden_states,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                ) = self.norm1(
                    hidden_states,
                    timestep,
                    class_labels,
                    hidden_dtype=hidden_states.dtype,
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # 1. Self-Attention
            # self.only_cross_attention = False
            cross_attention_kwargs = (
                cross_attention_kwargs if cross_attention_kwargs is not None else {}
            )
            if self.only_cross_attention:
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states
                    if self.only_cross_attention
                    else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            else:
                if MODE == "write":
                    active_bank = getattr(self, "active_bank", "global")
                    if not hasattr(self, "banks"):
                        self.banks = {"global": [], "upper": [], "lower": []}
                    self.banks.setdefault(active_bank, []).append(norm_hidden_states.clone())
                    if active_bank == "global":
                        self.bank = self.banks["global"]
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states
                        if self.only_cross_attention
                        else None,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                if MODE == "read":
                    def read_bank(bank_name: str):
                        banks = getattr(self, "banks", None)
                        if banks is None:
                            banks = {"global": getattr(self, "bank", [])}
                        bank_values = banks.get(bank_name, [])
                        if not bank_values:
                            if bank_name != "global":
                                return None
                            bank_values = getattr(self, "bank", [])
                        if bank_values:
                            bank_fea = [
                                rearrange(
                                    d.unsqueeze(1).repeat(1, video_length, 1, 1),
                                    "b t l c -> (b t) l c",
                                )
                                for d in bank_values
                            ]
                            encoder_states = torch.cat(
                                [norm_hidden_states] + bank_fea, dim=1
                            )
                        else:
                            encoder_states = norm_hidden_states
                        return self.attn1(
                            norm_hidden_states,
                            encoder_hidden_states=encoder_states,
                            attention_mask=attention_mask,
                        )

                    global_output = read_bank("global")
                    if (
                        controller.part_bank_enabled
                        and controller.part_bank_fusion is not None
                    ):
                        upper_output = read_bank("upper")
                        lower_output = read_bank("lower")
                        upper_mask = mask_to_token_weights(
                            getattr(self, "target_upper_mask", None),
                            norm_hidden_states.shape[1],
                            hidden_batch=norm_hidden_states.shape[0],
                            dtype=norm_hidden_states.dtype,
                            device=norm_hidden_states.device,
                        )
                        lower_mask = mask_to_token_weights(
                            getattr(self, "target_lower_mask", None),
                            norm_hidden_states.shape[1],
                            hidden_batch=norm_hidden_states.shape[0],
                            dtype=norm_hidden_states.dtype,
                            device=norm_hidden_states.device,
                        )
                        ref_output = controller.part_bank_fusion(
                            global_output,
                            upper_output,
                            lower_output,
                            upper_mask,
                            lower_mask,
                        )
                    else:
                        ref_output = global_output

                    hidden_states_uc = ref_output + hidden_states
                    if do_classifier_free_guidance:
                        hidden_states_c = hidden_states_uc.clone()
                        _uc_mask = uc_mask.clone()
                        if hidden_states.shape[0] != _uc_mask.shape[0]:
                            _uc_mask = (
                                torch.Tensor(
                                    [1] * (hidden_states.shape[0] // 2)
                                    + [0] * (hidden_states.shape[0] // 2)
                                )
                                .to(device)
                                .bool()
                            )
                        hidden_states_c[_uc_mask] = (
                            self.attn1(
                                norm_hidden_states[_uc_mask],
                                encoder_hidden_states=norm_hidden_states[_uc_mask],
                                attention_mask=attention_mask,
                            )
                            + hidden_states[_uc_mask]
                        )
                        hidden_states = hidden_states_c.clone()
                    else:
                        hidden_states = hidden_states_uc

                    # self.bank.clear()
                    if self.attn2 is not None:
                        # Cross-Attention
                        norm_hidden_states = (
                            self.norm2(hidden_states, timestep)
                            if self.use_ada_layer_norm
                            else self.norm2(hidden_states)
                        )
                        cross_output = self.attn2(
                            norm_hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            attention_mask=attention_mask,
                        )
                        if (
                            controller.color_structure_enabled
                            and controller.color_scale > 0
                            and getattr(self, "color_upper_states", None) is not None
                            and getattr(self, "color_lower_states", None) is not None
                        ):
                            color_upper_states = self.color_upper_states
                            color_lower_states = self.color_lower_states
                            if color_upper_states.shape[0] != norm_hidden_states.shape[0]:
                                if norm_hidden_states.shape[0] % color_upper_states.shape[0] == 0:
                                    repeat = norm_hidden_states.shape[0] // color_upper_states.shape[0]
                                    color_upper_states = color_upper_states.repeat(
                                        repeat, 1, 1
                                    )
                                    color_lower_states = color_lower_states.repeat(
                                        repeat, 1, 1
                                    )
                                else:
                                    color_upper_states = None
                                    color_lower_states = None
                            upper_mask = mask_to_token_weights(
                                getattr(self, "target_upper_mask", None),
                                norm_hidden_states.shape[1],
                                hidden_batch=norm_hidden_states.shape[0],
                                dtype=norm_hidden_states.dtype,
                                device=norm_hidden_states.device,
                            )
                            lower_mask = mask_to_token_weights(
                                getattr(self, "target_lower_mask", None),
                                norm_hidden_states.shape[1],
                                hidden_batch=norm_hidden_states.shape[0],
                                dtype=norm_hidden_states.dtype,
                                device=norm_hidden_states.device,
                            )
                            if (
                                upper_mask is not None
                                and lower_mask is not None
                                and color_upper_states is not None
                                and color_lower_states is not None
                            ):
                                upper_color = self.attn2(
                                    norm_hidden_states,
                                    encoder_hidden_states=color_upper_states,
                                    attention_mask=None,
                                )
                                lower_color = self.attn2(
                                    norm_hidden_states,
                                    encoder_hidden_states=color_lower_states,
                                    attention_mask=None,
                                )
                                cross_output = cross_output + controller.color_scale * (
                                    upper_mask * upper_color + lower_mask * lower_color
                                )
                        hidden_states = cross_output + hidden_states

                    # Feed-forward
                    hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

                    # Temporal-Attention
                    if self.unet_use_temporal_attention:
                        d = hidden_states.shape[1]
                        hidden_states = rearrange(
                            hidden_states, "(b f) d c -> (b d) f c", f=video_length
                        )
                        norm_hidden_states = (
                            self.norm_temp(hidden_states, timestep)
                            if self.use_ada_layer_norm
                            else self.norm_temp(hidden_states)
                        )
                        hidden_states = (
                            self.attn_temp(norm_hidden_states) + hidden_states
                        )
                        hidden_states = rearrange(
                            hidden_states, "(b d) f c -> (b f) d c", d=d
                        )

                    return hidden_states

            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep)
                    if self.use_ada_layer_norm
                    else self.norm2(hidden_states)
                )

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = (
                    norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                )

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states

        if self.reference_attn:
            if self.fusion_blocks == "midup":
                attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                    or isinstance(module, TemporalBasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, BasicTransformerBlock)
                    or isinstance(module, TemporalBasicTransformerBlock)
                ]
            attn_modules = sorted(
                attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )

            for i, module in enumerate(attn_modules):
                module._original_inner_forward = module.forward
                if isinstance(module, BasicTransformerBlock):
                    module.forward = hacked_basic_transformer_inner_forward.__get__(
                        module, BasicTransformerBlock
                    )
                if isinstance(module, TemporalBasicTransformerBlock):
                    module.forward = hacked_basic_transformer_inner_forward.__get__(
                        module, TemporalBasicTransformerBlock
                    )

                module.banks = {"global": [], "upper": [], "lower": []}
                module.active_bank = "global"
                module.bank = module.banks["global"]
                module.target_upper_mask = None
                module.target_lower_mask = None
                module.color_upper_states = None
                module.color_lower_states = None
                module.attn_weight = float(i) / float(len(attn_modules))

    def update(self, writer, dtype=torch.float16):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, TemporalBasicTransformerBlock)
                ]
                writer_attn_modules = [
                    module
                    for module in (
                        torch_dfs(writer.unet.mid_block)
                        + torch_dfs(writer.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, TemporalBasicTransformerBlock)
                ]
                writer_attn_modules = [
                    module
                    for module in torch_dfs(writer.unet)
                    if isinstance(module, BasicTransformerBlock)
                ]
            reader_attn_modules = sorted(
                reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            writer_attn_modules = sorted(
                writer_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            for r, w in zip(reader_attn_modules, writer_attn_modules):
                writer_banks = getattr(w, "banks", {"global": getattr(w, "bank", [])})
                r.banks = {
                    name: [v.clone().to(dtype) for v in writer_banks.get(name, [])]
                    for name in ("global", "upper", "lower")
                }
                r.bank = r.banks["global"]
                r.target_upper_mask = self.target_upper_mask
                r.target_lower_mask = self.target_lower_mask
                r.color_upper_states = self.color_upper_states
                r.color_lower_states = self.color_lower_states
                # w.bank.clear()

    def _registered_modules(self):
        if self.fusion_blocks == "midup":
            modules = torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
        elif self.fusion_blocks == "full":
            modules = torch_dfs(self.unet)
        else:
            modules = []
        modules = [
            module
            for module in modules
            if isinstance(module, BasicTransformerBlock)
            or isinstance(module, TemporalBasicTransformerBlock)
        ]
        return sorted(modules, key=lambda x: -x.norm1.normalized_shape[0])

    def set_active_bank(self, bank_name: str):
        if bank_name not in {"global", "upper", "lower"}:
            raise ValueError(f"Unsupported bank name: {bank_name}")
        for module in self._registered_modules():
            module.active_bank = bank_name

    def set_target_masks(
        self,
        upper_mask: Optional[torch.Tensor],
        lower_mask: Optional[torch.Tensor],
    ):
        self.target_upper_mask = upper_mask
        self.target_lower_mask = lower_mask
        for module in self._registered_modules():
            module.target_upper_mask = upper_mask
            module.target_lower_mask = lower_mask

    def set_color_states(
        self,
        upper_states: Optional[torch.Tensor],
        lower_states: Optional[torch.Tensor],
        color_scale: Optional[float] = None,
    ):
        self.color_upper_states = upper_states
        self.color_lower_states = lower_states
        if color_scale is not None:
            self.color_scale = float(color_scale)
        for module in self._registered_modules():
            module.color_upper_states = upper_states
            module.color_lower_states = lower_states

    def clear_conditions(self):
        self.set_target_masks(None, None)
        self.set_color_states(None, None)

    def clear(self):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                    or isinstance(module, TemporalBasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, BasicTransformerBlock)
                    or isinstance(module, TemporalBasicTransformerBlock)
                ]
            reader_attn_modules = sorted(
                reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            for r in reader_attn_modules:
                if hasattr(r, "banks"):
                    for bank_values in r.banks.values():
                        bank_values.clear()
                    r.bank = r.banks["global"]
                elif hasattr(r, "bank"):
                    r.bank.clear()
                r.target_upper_mask = None
                r.target_lower_mask = None
                r.color_upper_states = None
                r.color_lower_states = None
            self.target_upper_mask = None
            self.target_lower_mask = None
            self.color_upper_states = None
            self.color_lower_states = None
