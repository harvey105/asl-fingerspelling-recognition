import argparse


def parse_train_args() -> argparse.Namespace:
    """Parse command-line arguments for model training."""
    parser = argparse.ArgumentParser(
        description="ASL Fingerspelling Recognition Model Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="simple_cnn",
        choices=[
            "simple_cnn",
            "complex_cnn",
            "transfer_cnn",
            "model_1",
            "model_2",
            "model_3",
        ],
        help="Target architecture: model_1 (simple), model_2 (complex), model_3 (transfer)",
    )
    parser.add_argument(
        "--split-type",
        type=str,
        default="subject",
        choices=["subject", "random"],
        help="Data split partition: 'subject' (subject-independent) or 'random'",
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
        "--patience",
        type=int,
        default=8,
        help="Early stopping patience (epochs without validation F1 gain)",
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
        help="Target compute device ('dml', 'cpu', 'cuda', or None for auto)",
    )
    return parser.parse_args()
