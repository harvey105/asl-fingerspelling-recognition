import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from src.models import build_model
from src.utils.gradcam import GradCAM, overlay_heatmap


def load_model(model_name: str, checkpoint_path: Path | str, device: torch.device) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    if model_name == "simple_cnn":
        model = build_model("simple_cnn", num_classes=36, fc_dim=256)
    elif model_name == "complex_cnn":
        fc_dim = state["classifier.2.weight"].shape[0]
        model = build_model("complex_cnn", num_classes=36, fc_dim=fc_dim)
    elif model_name == "transfer_cnn":
        fc_dim = 0 if "model.fc.1.weight" in state and state["model.fc.1.weight"].shape[0] == 36 else 256
        model = build_model("transfer_cnn", num_classes=36, fc_dim=fc_dim, pretrained=False)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    if model_name == "simple_cnn":
        return model.features[12]
    elif model_name == "complex_cnn":
        return model.stages[7].conv2
    elif model_name == "transfer_cnn":
        return model.model.layer4[1].conv2
    raise ValueError(f"Unknown model name: {model_name}")


def preprocess_image(rgb_image: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    normalized = (tensor - mean) / std
    return normalized.unsqueeze(0).to(device)


def generate_attribution_evolution_panel(
    models: dict[str, nn.Module],
    device: torch.device,
    output_path: Path,
) -> None:
    cases = [
        {"class": "B", "id": 11, "path": "data/processed_crops/B/P9_B_801.png", "desc": "B (Flat Palm Blade)"},
        {"class": "O", "id": 24, "path": "data/processed_crops/O/P9_O_801.png", "desc": "O (Ovoid Ring Enclosure)"},
        {"class": "Y", "id": 34, "path": "data/processed_crops/Y/P9_Y_801.png", "desc": "Y (Radial/Ulnar Spread)"},
        {"class": "L", "id": 21, "path": "data/processed_crops/L/P9_L_801.png", "desc": "L (Orthogonal Index & Thumb)"},
    ]

    _, axes = plt.subplots(len(cases), 4, figsize=(16, 4.0 * len(cases)), dpi=200)

    for r, case in enumerate(cases):
        bgr = cv2.imread(case["path"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = preprocess_image(rgb, device)

        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(f"Input Crop\n{case['desc']}", fontsize=11, fontweight="bold")
        axes[r, 0].axis("off")

        for c, (m_key, m_label) in enumerate(
            [("simple_cnn", "Model 1: Simple CNN"), ("complex_cnn", "Model 2: Complex CNN"), ("transfer_cnn", "Model 3: Transfer CNN")],
            start=1,
        ):
            model = models[m_key]
            target_layer = get_target_layer(model, m_key)
            with GradCAM(model, target_layer) as cam:
                cam_map = cam(inp, target_class=case["id"]).cpu().numpy()[0]
                overlay = overlay_heatmap(rgb, cam_map, alpha=0.5)

            axes[r, c].imshow(overlay)
            axes[r, c].set_title(m_label if r == 0 else "", fontsize=11, fontweight="bold" if r == 0 else "normal")
            axes[r, c].axis("off")

    plt.suptitle("Architectural Attribution Evolution Across Convolutional Hierarchies", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def generate_counterfactual_diagnostic_panel(
    model3: nn.Module,
    device: torch.device,
    output_path: Path,
) -> None:
    pairs = [
        {
            "true_name": "6",
            "true_id": 6,
            "pred_name": "W",
            "pred_id": 32,
            "path": "data/processed_crops/6/P9_6_801.png",
            "conf": 0.270,
            "note": "Morphological ambiguity: 3 upright fingers with curled pinky/thumb contact",
        },
        {
            "true_name": "1",
            "true_id": 1,
            "pred_name": "Z",
            "pred_id": 35,
            "path": "data/processed_crops/1/P9_1_801.png",
            "conf": 0.427,
            "note": "Single index upright vs dynamic Z trace initial static pose",
        },
        {
            "true_name": "R",
            "true_id": 27,
            "pred_name": "U",
            "pred_id": 30,
            "path": "data/processed_crops/R/P10_R_901.png",
            "conf": 0.300,
            "note": "Intertwined fingers (R) vs adjacent parallel fingers (U)",
        },
    ]

    target_layer = get_target_layer(model3, "transfer_cnn")
    _, axes = plt.subplots(len(pairs), 3, figsize=(14, 4.3 * len(pairs)), dpi=200)

    for r, pair in enumerate(pairs):
        bgr = cv2.imread(pair["path"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = preprocess_image(rgb, device)

        with GradCAM(model3, target_layer) as cam:
            cam_pred = cam(inp, target_class=pair["pred_id"]).cpu().numpy()[0]
            cam_true = cam(inp, target_class=pair["true_id"]).cpu().numpy()[0]

        ov_pred = overlay_heatmap(rgb, cam_pred, alpha=0.5)
        ov_true = overlay_heatmap(rgb, cam_true, alpha=0.5)

        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(f"Crop (Ground Truth: '{pair['true_name']}')\n{pair['note']}", fontsize=10)
        axes[r, 0].axis("off")

        axes[r, 1].imshow(ov_pred)
        axes[r, 1].set_title(f"Targeting Mistake: Pred '{pair['pred_name']}'\nConfidence: {pair['conf']:.2f}", fontsize=10, fontweight="bold")
        axes[r, 1].axis("off")

        axes[r, 2].imshow(ov_true)
        axes[r, 2].set_title(f"Targeting Counterfactual: True '{pair['true_name']}'\n(Discriminative anatomical signal)", fontsize=10, fontweight="bold")
        axes[r, 2].axis("off")

    plt.suptitle("Counterfactual Diagnostic Attribution: Predicted vs Ground-Truth Logit Saliency", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def generate_letterbox_invariance_panel(
    models: dict[str, nn.Module],
    device: torch.device,
    output_path: Path,
) -> None:
    samples = [
        {"signer": "P9", "class": "B", "id": 11, "path": "data/processed_crops/B/P9_B_801.png", "condition": "Fair / Indoor Cool"},
        {"signer": "P10", "class": "B", "id": 11, "path": "data/processed_crops/B/P10_B_1000.png", "condition": "Deep Tone / High Contrast"},
        {"signer": "P9", "class": "Y", "id": 34, "path": "data/processed_crops/Y/P9_Y_801.png", "condition": "Wide Span / Lateral Padding"},
        {"signer": "P10", "class": "Y", "id": 34, "path": "data/processed_crops/Y/P10_Y_1000.png", "condition": "Deep Tone / Lateral Padding"},
    ]

    _, axes = plt.subplots(len(samples), 4, figsize=(16, 4.0 * len(samples)), dpi=200)

    for r, sample in enumerate(samples):
        bgr = cv2.imread(sample["path"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = preprocess_image(rgb, device)
        gray_mask = np.all(np.abs(bgr.astype(int) - 128) < 5, axis=-1)

        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(f"Signer {sample['signer']} - '{sample['class']}'\n{sample['condition']}", fontsize=11, fontweight="bold")
        axes[r, 0].axis("off")

        for c, (m_key, m_label) in enumerate(
            [("simple_cnn", "Model 1 (Simple CNN)"), ("complex_cnn", "Model 2 (Complex CNN)"), ("transfer_cnn", "Model 3 (Transfer CNN)")],
            start=1,
        ):
            model = models[m_key]
            target_layer = get_target_layer(model, m_key)
            with GradCAM(model, target_layer) as cam:
                cam_map = cam(inp, target_class=sample["id"]).cpu().numpy()[0]
                overlay = overlay_heatmap(rgb, cam_map, alpha=0.5)

            pad_salience = cam_map[gray_mask].mean() if gray_mask.sum() > 0 else 0.0
            hand_salience = cam_map[~gray_mask].mean() if (~gray_mask).sum() > 0 else 1.0
            ratio = pad_salience / (hand_salience + 1e-8)

            axes[r, c].imshow(overlay)
            axes[r, c].set_title(
                f"{m_label}\nPadding Leak: {ratio:.1%}",
                fontsize=10,
                color="red" if ratio > 0.4 else "darkgreen",
            )
            axes[r, c].axis("off")

    plt.suptitle("Letterbox Invariance & Artifact Suppression Across Diverse Signers", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visual interpretability panels using Grad-CAM.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "dml", "cuda"])
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures/interpretability"))
    args = parser.parse_args()

    if args.device == "auto":
        try:
            import torch_directml
            device = torch_directml.device()
        except ImportError:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "dml":
        import torch_directml
        device = torch_directml.device()
    else:
        device = torch.device(args.device)

    print(f"Running Grad-CAM generation on device: {device}")

    models = {
        "simple_cnn": load_model("simple_cnn", "weights/best_model_1_simple_cnn.pt", device),
        "complex_cnn": load_model("complex_cnn", "weights/best_model_2_complex_cnn.pt", device),
        "transfer_cnn": load_model("transfer_cnn", "weights/best_model_3_transfer_cnn.pt", device),
    }

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating architectural attribution evolution panel...")
    generate_attribution_evolution_panel(models, device, out_dir / "architectural_attribution_evolution.png")

    print("Generating counterfactual diagnostic panel...")
    generate_counterfactual_diagnostic_panel(models["transfer_cnn"], device, out_dir / "counterfactual_diagnostic_attribution.png")

    print("Generating letterbox invariance verification panel...")
    generate_letterbox_invariance_panel(models, device, out_dir / "letterbox_invariance_verification.png")

    print(f"All attribution panels successfully exported to {out_dir}")


if __name__ == "__main__":
    main()
