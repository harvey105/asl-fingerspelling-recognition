"""
Hardware throughput benchmark on DirectML / CPU devices.
"""

import argparse
from pathlib import Path
import sys
import time

# Ensure project root is discoverable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models import build_model
from src.training.engine import TrainerConfig, get_default_device
from src.training.losses import LabelSmoothingCrossEntropy


def benchmark_device(
    model_name: str = "simple_cnn",
    device_name: str = "dml",
    batch_size: int = 64,
    num_batches: int = 10,
    warmup_batches: int = 2,
) -> None:
    device = get_default_device(device_name)
    print(f"============================================================")
    print(f"Hardware Benchmark: {model_name} on {device}")
    if hasattr(torch, "directml") or "privateuseone" in str(device):
        try:
            import torch_directml

            print(f"DirectML Adapter: {torch_directml.device_name(0)}")
        except Exception:
            pass
    print(f"Batch size: {batch_size} | Total timed batches: {num_batches}")
    print(f"============================================================")

    model = build_model(model_name, num_classes=36).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    # Warmup
    print(f"Warming up ({warmup_batches} batches)...")
    for _ in range(warmup_batches):
        x = torch.randn(batch_size, 3, 224, 224, device=device)
        y = torch.randint(0, 36, (batch_size,), device=device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    print("Running timed iterations...")
    start_time = time.perf_counter()
    for _ in range(num_batches):
        x = torch.randn(batch_size, 3, 224, 224, device=device)
        y = torch.randint(0, 36, (batch_size,), device=device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    elapsed = time.perf_counter() - start_time
    total_images = batch_size * num_batches
    throughput = total_images / elapsed

    print(f"Completed {total_images} images in {elapsed:.2f}s")
    print(f"Throughput: {throughput:.2f} images/sec")
    print(f"Average batch latency: {(elapsed / num_batches) * 1000:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DirectML Hardware Benchmark")
    parser.add_argument("--model", type=str, default="simple_cnn")
    parser.add_argument("--device", type=str, default="dml")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=10)
    args = parser.parse_args()

    benchmark_device(
        model_name=args.model,
        device_name=args.device,
        batch_size=args.batch_size,
        num_batches=args.batches,
    )
