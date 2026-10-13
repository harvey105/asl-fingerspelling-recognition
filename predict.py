from pathlib import Path

import torch

from src.utils.predict_utils import (
    evaluate_model,
    load_model,
    normalize_model_name,
    parse_predict_args,
    predict_single_image,
    prepare_test_loader,
    save_and_display,
)

TEST_CSV = Path("data/splits/subject_test.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_predict_args()
    output_path = Path(args.output)
    selected_models = [normalize_model_name(m) for m in args.models]

    lines: list[str] = [
        "=" * 64,
        "   ASL FINGERSPELLING RECOGNITION - EVALUATION & INFERENCE",
        f"   Compute Device: {DEVICE}",
        "=" * 64,
        "",
    ]

    # 1. Single image prediction mode
    if args.image:
        image_path = Path(args.image)
        lines.append(f"Target Image: {image_path}\n")
        for model_key in selected_models:
            model, title = load_model(model_key, DEVICE)
            pred_label, conf = predict_single_image(model, image_path, DEVICE)
            lines.append(f"  {title:<35} -> Predicted: {pred_label:>2} (Conf: {conf:5.1f}%)")
        lines.append("")
        save_and_display(lines, output_path)
        return

    # 2. Test dataset evaluation mode (default: 250 sample images; --full: 7,200 images)
    loader, count = prepare_test_loader(
        TEST_CSV,
        full=args.full,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
    )
    mode_desc = "Full Test Set (7,200 images)" if args.full else f"Random Test Sample ({count} images)"
    lines.append(f"Evaluation Mode: {mode_desc}\n")

    summary_table = []
    for model_key in selected_models:
        model, title = load_model(model_key, DEVICE)
        acc, correct, total, previews = evaluate_model(model, loader, DEVICE)

        lines.extend((f"--- {title} ---", "Sample Predictions (Randomized Subset):"))
        lines.extend(previews)
        lines.append(f"Result: {correct}/{total} correct -> Accuracy: {acc:6.2f}%\n")
        summary_table.append((title, f"{acc:6.2f}%", f"{correct}/{total}"))

    lines.extend(("=" * 64, "   EVALUATION SUMMARY", "=" * 64))
    for title, acc, ratio in summary_table:
        lines.append(f"  {title:<38} {acc:>8}  ({ratio})")
    lines.append("=" * 64)

    save_and_display(lines, output_path)


if __name__ == "__main__":
    main()
