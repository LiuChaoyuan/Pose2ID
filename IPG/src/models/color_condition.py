from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class ColorTextEncoder(nn.Module):
    """Frozen CLIP text encoder for upper/lower clothing color descriptions."""

    def __init__(
        self,
        pretrained_model_path: str,
        *,
        tokenizer_subfolder: str = "tokenizer",
        text_encoder_subfolder: str = "text_encoder",
        max_length: int = None,
    ):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_path, subfolder=tokenizer_subfolder
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_path, subfolder=text_encoder_subfolder
        )
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.max_length = max_length or self.tokenizer.model_max_length

    @property
    def dtype(self):
        return self.text_encoder.dtype

    def _encode_texts(
        self,
        texts: Sequence[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        clean_texts: List[str] = []
        missing = []
        for text in texts:
            text = text if isinstance(text, str) else ""
            text = text.strip()
            missing.append(text == "")
            clean_texts.append(text if text else " ")

        inputs = self.tokenizer(
            clean_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(device=device) for key, value in inputs.items()}
        hidden_states = self.text_encoder(**inputs).last_hidden_state.to(dtype=dtype)
        if any(missing):
            missing_mask = torch.tensor(missing, device=device, dtype=torch.bool)
            hidden_states = hidden_states.clone()
            hidden_states[missing_mask] = 0
        return hidden_states

    @torch.no_grad()
    def encode_pair(
        self,
        upper_texts: Sequence[str],
        lower_texts: Sequence[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        upper = self._encode_texts(upper_texts, device=device, dtype=dtype)
        lower = self._encode_texts(lower_texts, device=device, dtype=dtype)
        return upper, lower
