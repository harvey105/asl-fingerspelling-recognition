"""
tests/test_env.py
Verifies library imports, versions, and device fallback.
"""

import sys

def run_smoke_test():
    print("=" * 60)
    print("ASL Fingerspelling: Environment Smoke Test")
    print("=" * 60)

    # 1. Check Python
    print(f"[OK] Python Version: {sys.version.split()[0]}")
    assert sys.version_info >= (3, 10), "Requires Python 3.10+"

    # 2. Check PyTorch
    import torch
    import torchvision
    print(f"[OK] PyTorch: {torch.__version__}")
    print(f"[OK] Torchvision: {torchvision.__version__}")

    # Device check (GPU or CPU fallback)
    if torch.cuda.is_available():
        print(f"[OK] Device: CUDA ({torch.cuda.get_device_name(0)})")
        device = "cuda"
    else:
        print("[INFO] Device: CPU (Fallback active)")
        device = "cpu"

    # Tensor math check
    dummy = torch.ones((2, 3, 200, 200), device=device) * 5
    assert dummy.sum().item() > 0, "Tensor calculation error!"
    print(f"[OK] Tensor allocation & arithmetic passed on {device.upper()}")

    # 3. Check Computer Vision & Processing
    import cv2
    import mediapipe as mp
    import albumentations as A
    import PIL
    import numpy as np
    import pandas as pd
    import sklearn
    print(f"[OK] OpenCV: {cv2.__version__}")
    print(f"[OK] MediaPipe: {mp.__version__}")
    print(f"[OK] Albumentations: {A.__version__}")
    print(f"[OK] NumPy: {np.__version__}")
    print(f"[OK] Pandas: {pd.__version__}")
    print(f"[OK] Scikit-Learn: {sklearn.__version__}")

    # 4. Check UI
    import gradio as gr
    print(f"[OK] Gradio: {gr.__version__}")

    print("=" * 60)
    print("ALL ENVIRONMENT CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_smoke_test()