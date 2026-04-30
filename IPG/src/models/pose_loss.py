from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiablePoseLoss(nn.Module):
    """A lightweight differentiable proxy for pose-distance supervision.

    This does not run an external pose detector. It estimates x0 at low-noise
    timesteps, decodes it through the frozen VAE, extracts soft image edges, and
    compares their soft distance field to the target pose image distance field.
    """

    def __init__(
        self,
        *,
        weight: float = 0.05,
        max_sigma: float = 0.35,
        distance_iterations: int = 8,
        edge_temperature: float = 8.0,
    ):
        super().__init__()
        self.weight = float(weight)
        self.max_sigma = float(max_sigma)
        self.distance_iterations = int(distance_iterations)
        self.edge_temperature = float(edge_temperature)
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def _predict_x0(
        self,
        model_pred: torch.Tensor,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        scheduler,
        prediction_type: str,
    ) -> torch.Tensor:
        if model_pred.ndim == 5:
            model_pred = model_pred.squeeze(2)
        if noisy_latents.ndim == 5:
            noisy_latents = noisy_latents.squeeze(2)

        alphas_cumprod = scheduler.alphas_cumprod.to(
            device=timesteps.device, dtype=noisy_latents.dtype
        )
        alpha = alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1)
        sigma = (1.0 - alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)

        if prediction_type == "epsilon":
            return (noisy_latents - sigma * model_pred) / alpha.clamp_min(1e-6)
        if prediction_type == "v_prediction":
            return alpha * noisy_latents - sigma * model_pred
        raise ValueError(f"Unsupported prediction type: {prediction_type}")

    def _soft_edges(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 3:
            gray = (
                0.299 * image[:, 0:1]
                + 0.587 * image[:, 1:2]
                + 0.114 * image[:, 2:3]
            )
        else:
            gray = image[:, :1]
        sobel_x = self.sobel_x.to(device=image.device, dtype=image.dtype)
        sobel_y = self.sobel_y.to(device=image.device, dtype=image.dtype)
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
        return torch.sigmoid(self.edge_temperature * (magnitude - magnitude.mean(dim=(2, 3), keepdim=True)))

    def _soft_distance_field(self, occupancy: torch.Tensor) -> torch.Tensor:
        occupancy = occupancy.clamp(0, 1)
        reach = occupancy
        accum = occupancy
        for _ in range(max(self.distance_iterations, 1)):
            reach = F.max_pool2d(reach, kernel_size=3, stride=1, padding=1)
            accum = accum + reach
        return 1.0 - (accum / float(max(self.distance_iterations, 1) + 1)).clamp(0, 1)

    def _pose_occupancy(self, pose_images: torch.Tensor, target_size) -> torch.Tensor:
        pose = pose_images.float()
        if pose.min() < 0:
            pose = (pose + 1.0) / 2.0
        occupancy = pose.abs().amax(dim=1, keepdim=True).clamp(0, 1)
        occupancy = F.interpolate(
            occupancy, size=target_size, mode="bilinear", align_corners=False
        )
        return occupancy

    def forward(
        self,
        *,
        model_pred: torch.Tensor,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        scheduler,
        vae,
        pose_images: torch.Tensor,
        prediction_type: str,
        latent_scale: float,
        low_noise_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.weight <= 0:
            return model_pred.new_zeros(())

        alphas_cumprod = scheduler.alphas_cumprod.to(
            device=timesteps.device, dtype=model_pred.dtype
        )
        sigma = (1.0 - alphas_cumprod[timesteps]).sqrt()
        selected = sigma <= self.max_sigma
        if low_noise_mask is not None:
            selected = selected & low_noise_mask.to(device=selected.device, dtype=torch.bool)
        if not selected.any():
            return model_pred.new_zeros(())

        x0_latents = self._predict_x0(
            model_pred=model_pred,
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            scheduler=scheduler,
            prediction_type=prediction_type,
        )
        x0_latents = x0_latents[selected] / float(latent_scale)
        pose_images = pose_images[selected]

        decoded = vae.decode(x0_latents.to(dtype=vae.dtype)).sample
        decoded = (decoded.float() / 2.0 + 0.5).clamp(0, 1)
        gen_edges = self._soft_edges(decoded)
        gen_distance = self._soft_distance_field(gen_edges)

        target_occ = self._pose_occupancy(pose_images, decoded.shape[-2:])
        target_distance = self._soft_distance_field(target_occ.to(decoded.dtype))
        return F.mse_loss(gen_distance, target_distance, reduction="mean")
