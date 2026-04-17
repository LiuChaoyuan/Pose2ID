import torch
import torch.nn as nn
from einops import rearrange


class IFR(nn.Module):
    def __init__(self, input_dim: int = 3840, num_tokens: int = 20, hidden_dim: int = 768):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.proj_motion = nn.Linear(input_dim, num_tokens * hidden_dim)
        self.norm_motion = nn.LayerNorm(hidden_dim)

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        encoder_hidden_states = self.proj_motion(encoder_hidden_states)
        encoder_hidden_states = rearrange(
            encoder_hidden_states, "b (n d) -> b n d", n=self.num_tokens
        )
        encoder_hidden_states = self.norm_motion(encoder_hidden_states)
        return encoder_hidden_states
