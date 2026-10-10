from collections.abc import Sequence
from typing import Self

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(
            module: nn.Module,
            input_tensor: tuple[torch.Tensor, ...],
            output_tensor: torch.Tensor,
        ) -> None:
            self.activations = output_tensor.detach()

        def backward_hook(
            module: nn.Module,
            grad_input: tuple[torch.Tensor, ...],
            grad_output: tuple[torch.Tensor, ...],
        ) -> None:
            self.gradients = grad_output[0].detach()

        h_fwd = self.target_layer.register_forward_hook(forward_hook)
        h_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        self.handles.extend([h_fwd, h_bwd])

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.remove_hooks()

    def __call__(
        self,
        inputs: torch.Tensor,
        target_class: int | Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.model.eval()
        self.model.zero_grad()

        device = next(self.model.parameters()).device
        x = inputs.to(device)
        batch_size, _, height, width = x.shape

        logits = self.model(x)

        if target_class is None:
            targets = torch.argmax(logits, dim=1)
        elif isinstance(target_class, int):
            targets = torch.tensor([target_class] * batch_size, device=device)
        elif isinstance(target_class, (torch.Tensor, Sequence)):
            targets = torch.as_tensor(target_class, device=device, dtype=torch.long)
        else:
            raise TypeError(f"Unsupported target_class type: {type(target_class)}")

        score = logits.gather(1, targets.view(-1, 1)).sum()
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Activations or gradients not captured. Check target_layer connection.")

        alpha = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(alpha * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam_resized = F.interpolate(
            cam,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        b_min = cam_resized.amin(dim=(2, 3), keepdim=True)
        b_max = cam_resized.amax(dim=(2, 3), keepdim=True)
        cam_norm = (cam_resized - b_min) / (b_max - b_min + 1e-8)

        return cam_norm.squeeze(1)


def overlay_heatmap(
    rgb_image: np.ndarray,
    cam_map: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    if rgb_image.dtype != np.uint8:
        if rgb_image.max() <= 1.0:
            rgb_u8 = (rgb_image * 255).astype(np.uint8)
        else:
            rgb_u8 = rgb_image.astype(np.uint8)
    else:
        rgb_u8 = rgb_image.copy()

    cam_u8 = np.uint8(255 * np.clip(cam_map, 0.0, 1.0))
    heatmap_bgr = cv2.applyColorMap(cam_u8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(rgb_u8, 1.0 - alpha, heatmap_rgb, alpha, 0)
    return blended
