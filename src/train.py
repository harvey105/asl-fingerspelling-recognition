import sys
from pathlib import Path

# Ensure project root is discoverable when executing directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Unbuffer stdout and stderr immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from src.config import NUM_CLASSES, SPLITS_DIR
from src.data.dataloaders import get_dataloaders
from src.models import build_model
from src.training.engine import Trainer, TrainerConfig, get_default_device
from src.utils.train_utils import parse_train_args


def main() -> None:
    """Execute training run according to CLI arguments."""
    args = parse_train_args()
    device = get_default_device(args.device)
    print(f"Executing on device: {device}", flush=True)

    # Resolve dataset splits
    train_csv = SPLITS_DIR / f"{args.split_type}_train.csv"
    val_csv = SPLITS_DIR / f"{args.split_type}_val.csv"
    test_csv = SPLITS_DIR / f"{args.split_type}_test.csv"

    if not train_csv.exists():
        print(f"Split manifest not found: {train_csv}. Exiting.", flush=True)
        sys.exit(1)

    print(f"Loading {args.split_type} splits from {SPLITS_DIR}...", flush=True)
    train_loader, val_loader, test_loader = get_dataloaders(
        train=train_csv,
        val=val_csv,
        test=test_csv,
        batch_size=args.batch_size,
    )

    # Build model architecture
    model = build_model(args.model, num_classes=NUM_CLASSES, dropout=args.dropout)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model: {args.model} | Trainable parameters: {param_count:,}",
        flush=True,
    )

    # Configure training engine
    config = TrainerConfig(
        model_name=args.model,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        checkpoint_name=args.checkpoint_name,
        early_stopping_patience=args.patience,
        lr_backbone=args.lr_backbone,
    )

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
    )

    print(f"Starting training run ({args.epochs} epochs)...", flush=True)
    trainer.fit()
    print("Training finished.", flush=True)


if __name__ == "__main__":
    main()
