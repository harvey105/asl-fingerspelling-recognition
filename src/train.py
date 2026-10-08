"""
CLI entry point for model training and evaluation.
"""

import argparse
from pathlib import Path
import sys

# Ensure project root is discoverable when executing directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import CROPS_DIR, NUM_CLASSES, SPLITS_DIR
from src.data.dataloaders import get_dataloaders
from src.models import build_model
from src.training.engine import Trainer, TrainerConfig, get_default_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASL Fingerspelling Recognition Model Trainer")
    parser.add_argument(
        "--model",
        type=str,
        default="simple_cnn",
        choices=["simple_cnn", "complex_cnn", "transfer_cnn", "mobilenet", "resnet18"],
        help="Target neural network architecture",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor (0.0 for standard CE)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target execution device ('dml', 'cpu', 'cuda', or None for auto)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute 1 mock batch step to verify pipeline sanity without full training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_default_device(args.device)
    print(f"Executing on device: {device}")

    model = build_model(args.model, num_classes=NUM_CLASSES)
    print(f"Loaded architecture: {args.model}")

    config = TrainerConfig(
        model_name=args.model,
        epochs=1 if args.dry_run else args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
    )

    if args.dry_run:
        print("Dry run requested: constructing synthetic batches for single-step verification.")
        mock_inputs = torch.randn(args.batch_size, 3, 224, 224)
        mock_targets = torch.randint(0, NUM_CLASSES, (args.batch_size,))
        mock_ds = TensorDataset(mock_inputs, mock_targets)
        train_loader = DataLoader(mock_ds, batch_size=args.batch_size)
        val_loader = DataLoader(mock_ds, batch_size=args.batch_size)
    else:
        train_manifest = SPLITS_DIR / "train.csv"
        val_manifest = SPLITS_DIR / "val.csv"
        test_manifest = SPLITS_DIR / "test.csv"

        if not train_manifest.exists():
            print(f"Split manifests not found under {SPLITS_DIR}. Exiting.")
            sys.exit(1)

        train_loader, val_loader, _ = get_dataloaders(
            train=train_manifest,
            val=val_manifest,
            test=test_manifest,
            batch_size=args.batch_size,
        )

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    if args.dry_run:
        train_metrics = trainer.train_epoch()
        eval_metrics = trainer.evaluate(val_loader)
        print("Dry run training step completed successfully.")
        print(f"Train metrics: {train_metrics}")
        print(f"Val metrics: {eval_metrics}")
    else:
        print(f"Starting training run for {args.epochs} epochs...")
        trainer.fit()
        print("Training completed.")


if __name__ == "__main__":
    main()
