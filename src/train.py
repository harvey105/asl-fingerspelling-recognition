"""
CLI entry point for model training and evaluation.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is discoverable when executing directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Unbuffer stdout and stderr immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import NUM_CLASSES, SPLITS_DIR
from src.data.dataloaders import get_dataloaders
from src.models import build_model
from src.training.engine import Trainer, TrainerConfig, get_default_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ASL Fingerspelling Recognition Model Trainer"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="simple_cnn",
        choices=[
            "simple_cnn",
            "complex_cnn",
            "transfer_cnn",
            "mobilenet",
            "mobilenet_v3_large",
            "mobilenet_v3_small",
            "resnet18",
        ],
        help="Target neural network architecture",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "mobilenet_v3_large", "mobilenet_v3_small"],
        help="Pretrained vision backbone for transfer learning",
    )
    parser.add_argument(
        "--split-type",
        type=str,
        default="subject",
        choices=["subject", "random"],
        help="Data split partition: 'subject' (subject-independent) or 'random'",
    )
    parser.add_argument(
        "--train-csv", type=str, default=None, help="Custom train CSV path"
    )
    parser.add_argument("--val-csv", type=str, default=None, help="Custom val CSV path")
    parser.add_argument(
        "--test-csv", type=str, default=None, help="Custom test CSV path"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per step"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "--lr-backbone",
        type=float,
        default=None,
        help="Differential learning rate for backbone (e.g. 1e-4)",
    )
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=0,
        help="Number of initial epochs to keep backbone frozen",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="AdamW weight decay"
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor (0.0 for standard CE)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.3, help="Dropout probability"
    )
    parser.add_argument(
        "--fc-dim", type=int, default=256, help="Fully connected hidden dimension"
    )
    parser.add_argument(
        "--channels",
        type=str,
        default="32,64,128,256",
        help="Comma-separated feature channel dimensions",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=None,
        help="Custom checkpoint name (without .pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target execution device ('dml', 'cpu', 'cuda', or None for auto)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Early stopping patience (number of epochs without validation F1 improvement)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and run evaluation on test set using specified checkpoint",
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
    print(f"Executing on device: {device}", flush=True)

    channel_list = [int(c.strip()) for c in args.channels.split(",") if c.strip()]
    model_kwargs = {"dropout": args.dropout}
    if args.model in ("simple_cnn", "simple"):
        model_kwargs["channel_list"] = channel_list
        model_kwargs["fc_dim"] = args.fc_dim
    elif args.model in ("complex_cnn", "complex"):
        model_kwargs["stage_channels"] = channel_list
        model_kwargs["fc_dim"] = args.fc_dim
    elif args.model in (
        "transfer_cnn",
        "transfer",
        "resnet18",
        "mobilenet",
        "mobilenet_v3_large",
        "mobilenet_v3_small",
    ):
        model_kwargs["backbone_name"] = args.backbone
        model_kwargs["fc_dim"] = args.fc_dim

    model = build_model(args.model, num_classes=NUM_CLASSES, **model_kwargs)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Loaded architecture: {args.model} | Trainable parameters: {param_count:,}",
        flush=True,
    )

    config = TrainerConfig(
        model_name=args.model,
        epochs=1 if args.dry_run else args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        checkpoint_name=args.checkpoint_name,
        early_stopping_patience=args.patience,
        lr_backbone=args.lr_backbone,
        freeze_epochs=args.freeze_epochs,
    )

    if args.dry_run:
        print(
            "Dry run requested: constructing synthetic batches for single-step verification.",
            flush=True,
        )
        mock_inputs = torch.randn(args.batch_size, 3, 224, 224)
        mock_targets = torch.randint(0, NUM_CLASSES, (args.batch_size,))
        mock_ds = TensorDataset(mock_inputs, mock_targets)
        train_loader = DataLoader(mock_ds, batch_size=args.batch_size)
        val_loader = DataLoader(mock_ds, batch_size=args.batch_size)
        test_loader = DataLoader(mock_ds, batch_size=args.batch_size)
    else:
        prefix = args.split_type
        train_path = (
            Path(args.train_csv)
            if args.train_csv
            else (SPLITS_DIR / f"{prefix}_train.csv")
        )
        val_path = (
            Path(args.val_csv) if args.val_csv else (SPLITS_DIR / f"{prefix}_val.csv")
        )
        test_path = (
            Path(args.test_csv)
            if args.test_csv
            else (SPLITS_DIR / f"{prefix}_test.csv")
        )

        if not train_path.exists():
            print(f"Split manifest not found: {train_path}. Exiting.", flush=True)
            sys.exit(1)

        print(
            f"Loading data splits ({args.split_type}):\n  Train: {train_path}\n  Val:   {val_path}\n  Test:  {test_path}",
            flush=True,
        )
        train_loader, val_loader, test_loader = get_dataloaders(
            train=train_path,
            val=val_path,
            test=test_path,
            batch_size=args.batch_size,
        )

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
    )

    if args.dry_run:
        train_metrics = trainer.train_epoch()
        eval_metrics = trainer.evaluate(val_loader)
        print("Dry run training step completed successfully.", flush=True)
        print(f"Train metrics: {train_metrics}", flush=True)
        print(f"Val metrics: {eval_metrics}", flush=True)
    elif args.eval_only:
        ckpt_name = args.checkpoint_name or f"best_{args.model}"
        ckpt_path = Path("weights") / f"{ckpt_name}.pt"
        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path}", flush=True)
            sys.exit(1)
        print(f"Loading checkpoint for evaluation: {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        print("Evaluating on unseen test set...", flush=True)
        test_metrics = trainer.evaluate(test_loader)
        print(
            f"============================================================\n"
            f"Test Set Evaluation ({ckpt_path.name})\n"
            f"Top-1 Accuracy:  {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)\n"
            f"Macro-Precision: {test_metrics['precision_macro']:.4f}\n"
            f"Macro-Recall:    {test_metrics['recall_macro']:.4f}\n"
            f"Macro-F1:        {test_metrics['f1_macro']:.4f}\n"
            f"Loss:            {test_metrics['loss']:.4f}\n"
            f"============================================================",
            flush=True,
        )
    else:
        print(f"Starting training run for {args.epochs} epochs...", flush=True)
        trainer.fit()
        print("Training completed.", flush=True)


if __name__ == "__main__":
    main()
