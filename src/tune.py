"""
Hyperparameter optimization using Optuna with TPE sampling and Median pruning.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Unbuffer stdout and stderr immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Ensure project root is discoverable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import optuna

from src.config import NUM_CLASSES, SPLITS_DIR
from src.data.dataloaders import get_dataloaders
from src.models import build_model
from src.training.engine import Trainer, TrainerConfig, get_default_device


def create_objective(
    split_type: str = "subject",
    epochs_per_trial: int = 4,
    batch_size: int = 64,
    device_name: str | None = None,
):
    """Factory creating the Optuna objective function for architecture exploration."""
    device = get_default_device(device_name)
    train_path = SPLITS_DIR / f"{split_type}_train.csv"
    val_path = SPLITS_DIR / f"{split_type}_val.csv"
    test_path = SPLITS_DIR / f"{split_type}_test.csv"

    train_loader, val_loader, _ = get_dataloaders(
        train=train_path,
        val=val_path,
        test=test_path,
        batch_size=batch_size,
    )

    def objective(trial: optuna.Trial) -> float:
        # Sample architecture & optimization hyperparameters
        lr = trial.suggest_float("lr", 3e-4, 2e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 5e-4, log=True)
        dropout = trial.suggest_float("dropout", 0.2, 0.4, step=0.1)
        fc_dim = trial.suggest_categorical("fc_dim", [128, 256, 384])
        channels_str = trial.suggest_categorical(
            "channels",
            ["32,64,128,256", "32,64,128,128", "32,64,96,128"],
        )
        channel_list = [int(c.strip()) for c in channels_str.split(",")]
        label_smoothing = trial.suggest_categorical("label_smoothing", [0.05, 0.1])

        print(
            f"\n[Trial {trial.number}] Hyperparameters: lr={lr:.2e}, wd={weight_decay:.2e}, "
            f"dropout={dropout}, fc_dim={fc_dim}, channels={channels_str}, ls={label_smoothing}",
            flush=True,
        )

        model = build_model(
            "simple_cnn",
            num_classes=NUM_CLASSES,
            channel_list=channel_list,
            fc_dim=fc_dim,
            dropout=dropout,
        )
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Trial {trial.number}] Parameter count: {param_count:,}", flush=True)

        config = TrainerConfig(
            model_name="simple_cnn",
            epochs=epochs_per_trial,
            learning_rate=lr,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
            warmup_epochs=1,
            print_freq=100,
        )

        trainer = Trainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )

        best_trial_val_f1 = 0.0

        for epoch in range(1, epochs_per_trial + 1):
            train_metrics = trainer.train_epoch(
                epoch=epoch, total_epochs=epochs_per_trial
            )
            val_metrics = trainer.evaluate(val_loader)
            trainer.scheduler.step()

            val_f1 = val_metrics["f1_macro"]
            val_acc = val_metrics["accuracy"]
            best_trial_val_f1 = max(best_trial_val_f1, val_f1)

            print(
                f"[Trial {trial.number} | Epoch {epoch}/{epochs_per_trial}] "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}",
                flush=True,
            )

            # Report intermediate validation metric for pruning
            trial.report(val_f1, epoch)
            if trial.should_prune():
                print(
                    f"[Trial {trial.number}] PRUNED at epoch {epoch} (Val F1: {val_f1:.4f})",
                    flush=True,
                )
                raise optuna.TrialPruned()

        return best_trial_val_f1

    return objective


def run_study(
    num_trials: int = 6,
    epochs_per_trial: int = 4,
    split_type: str = "subject",
    device_name: str | None = None,
) -> optuna.Study:
    """Run Optuna hyperparameter study with TPESampler and MedianPruner."""
    log_dir = PROJECT_ROOT / "logs" / "tuning"
    log_dir.mkdir(parents=True, exist_ok=True)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=2,
        n_warmup_steps=1,
        interval_steps=1,
    )
    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="simple_cnn_hparams",
    )

    objective_fn = create_objective(
        split_type=split_type,
        epochs_per_trial=epochs_per_trial,
        device_name=device_name,
    )

    print(
        f"Starting Optuna Study ({num_trials} trials, max {epochs_per_trial} epochs/trial) "
        f"on {split_type} split...",
        flush=True,
    )
    study.optimize(objective_fn, n_trials=num_trials)

    print("\n============================================================")
    print("Optuna Hyperparameter Tuning Complete")
    print(f"Best Trial #{study.best_trial.number} with Val F1: {study.best_value:.4f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("============================================================\n")

    # Save summary
    study_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": "simple_cnn",
        "split_type": split_type,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }
    summary_path = log_dir / "optuna_study_simple_cnn.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(study_summary, f, indent=2)
    print(f"Study results saved to: {summary_path}", flush=True)

    return study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning")
    parser.add_argument("--trials", type=int, default=6, help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=4, help="Max epochs per trial")
    parser.add_argument(
        "--split-type", type=str, default="subject", choices=["subject", "random"]
    )
    parser.add_argument("--device", type=str, default=None, help="Execution device")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_study(
        num_trials=args.trials,
        epochs_per_trial=args.epochs,
        split_type=args.split_type,
        device_name=args.device,
    )
