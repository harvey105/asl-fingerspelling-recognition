import numpy as np
import torch
from torch import nn

from src.models.complex_cnn import ComplexCNN
from src.models.simple_cnn import SimpleCNN
from src.models.transfer_cnn import TransferCNN
from src.utils.gradcam import GradCAM, overlay_heatmap


class ToyConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def test_gradcam_output_shape_and_bounds() -> None:
    model = ToyConvNet()
    with GradCAM(model=model, target_layer=model.conv2) as cam_engine:
        inputs = torch.randn(2, 3, 32, 32)
        cams = cam_engine(inputs, target_class=2)

        assert cams.shape == (2, 32, 32)
        assert not torch.isnan(cams).any()
        assert not torch.isinf(cams).any()
        assert (cams >= 0.0).all()
        assert (cams <= 1.0).all()


def test_gradcam_overlay_dimensions() -> None:
    dummy_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    dummy_cam = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
    overlay = overlay_heatmap(dummy_rgb, dummy_cam, alpha=0.5)

    assert overlay.shape == (64, 64, 3)
    assert overlay.dtype == np.uint8


def test_gradcam_simple_cnn() -> None:
    model = SimpleCNN(num_classes=36)
    target_layer = model.features[12]
    with GradCAM(model=model, target_layer=target_layer) as cam_engine:
        x = torch.randn(1, 3, 224, 224)
        cams = cam_engine(x, target_class=5)
        assert cams.shape == (1, 224, 224)
        assert not torch.isnan(cams).any()
        assert (cams >= 0.0).all() and (cams <= 1.0).all()


def test_gradcam_complex_cnn() -> None:
    model = ComplexCNN(num_classes=36)
    target_layer = model.stages[7].conv2
    with GradCAM(model=model, target_layer=target_layer) as cam_engine:
        x = torch.randn(1, 3, 224, 224)
        cams = cam_engine(x, target_class=12)
        assert cams.shape == (1, 224, 224)
        assert not torch.isnan(cams).any()
        assert (cams >= 0.0).all() and (cams <= 1.0).all()


def test_gradcam_transfer_cnn() -> None:
    model = TransferCNN(num_classes=36, pretrained=False)
    target_layer = model.model.layer4[1].conv2
    with GradCAM(model=model, target_layer=target_layer) as cam_engine:
        x = torch.randn(1, 3, 224, 224)
        cams = cam_engine(x, target_class=20)
        assert cams.shape == (1, 224, 224)
        assert not torch.isnan(cams).any()
        assert (cams >= 0.0).all() and (cams <= 1.0).all()
