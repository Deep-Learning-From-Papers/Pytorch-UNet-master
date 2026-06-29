import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet import UNet
from utils.data_loading import BasicDataset
from utils.segmentation_metrics import binary_segmentation_metrics


# ----------------------------
# Configuration
# ----------------------------
IMG_DIR = Path("/content/brain_tumor_unet/valid/imgs")
MASK_DIR = Path("/content/brain_tumor_unet/valid/masks")
CHECKPOINT = Path("checkpoints/checkpoint_epoch10.pth")

RESULTS_DIR = Path("results")
PRED_DIR = RESULTS_DIR / "predictions"

SCALE = 0.5
BATCH_SIZE = 4
THRESHOLD = 0.5
N_EXAMPLES = 5
SEED = 42


# ----------------------------
# Setup
# ----------------------------
RESULTS_DIR.mkdir(exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
random.seed(SEED)


def load_model(device):
    model = UNet(n_channels=3, n_classes=1, bilinear=False)

    state_dict = torch.load(CHECKPOINT, map_location=device)
    state_dict.pop("mask_values", None)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def evaluate_model(model, dataloader, device):
    all_metrics = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device=device, dtype=torch.float32)
            masks = batch["mask"].to(device=device, dtype=torch.float32)

            logits = model(images)
            probs = torch.sigmoid(logits).squeeze(1)

            metrics = binary_segmentation_metrics(
                pred=probs,
                target=masks,
                threshold=THRESHOLD,
            )

            all_metrics.append(metrics)

    avg_metrics = {
        key: float(np.mean([m[key] for m in all_metrics]))
        for key in all_metrics[0]
    }

    return avg_metrics


def save_metrics(metrics):
    csv_path = RESULTS_DIR / "baseline_metrics.csv"
    txt_path = RESULTS_DIR / "baseline_metrics.txt"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])

    with open(txt_path, "w") as f:
        f.write("Baseline U-Net Validation Metrics\n")
        f.write("---------------------------------\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")


def print_metrics(metrics):
    print("\n===================================")
    print("BASELINE U-NET VALIDATION RESULTS")
    print("===================================")
    print(f"Dice      : {metrics['dice']:.4f}")
    print(f"IoU       : {metrics['iou']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1']:.4f}")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"TP        : {metrics['tp']:.0f}")
    print(f"TN        : {metrics['tn']:.0f}")
    print(f"FP        : {metrics['fp']:.0f}")
    print(f"FN        : {metrics['fn']:.0f}")
    print("===================================")


def plot_metrics(metrics):
    selected_metrics = {
        "Dice": metrics["dice"],
        "IoU": metrics["iou"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1": metrics["f1"],
        "Accuracy": metrics["accuracy"],
    }

    names = list(selected_metrics.keys())
    values = list(selected_metrics.values())

    plt.figure(figsize=(9, 5))
    plt.bar(names, values)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Baseline U-Net Validation Metrics")
    plt.grid(axis="y", alpha=0.3)

    out_path = RESULTS_DIR / "baseline_metrics_barplot.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


def save_prediction_examples(model, dataset, device):
    indices = random.sample(range(len(dataset)), min(N_EXAMPLES, len(dataset)))

    for i, idx in enumerate(indices, start=1):
        sample = dataset[idx]

        image_tensor = sample["image"]
        true_mask = sample["mask"].numpy()

        image = image_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            logits = model(image)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_mask = (prob > THRESHOLD).astype(np.uint8)

        image_np = image_tensor.permute(1, 2, 0).numpy()

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(image_np)
        plt.title("MRI Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(true_mask, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask, cmap="gray")
        plt.title("Prediction")
        plt.axis("off")

        plt.tight_layout()

        out_path = PRED_DIR / f"prediction_example_{i}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = BasicDataset(
        images_dir=IMG_DIR,
        mask_dir=MASK_DIR,
        scale=SCALE,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = load_model(device)

    metrics = evaluate_model(model, dataloader, device)

    save_metrics(metrics)
    print_metrics(metrics)
    plot_metrics(metrics)
    save_prediction_examples(model, dataset, device)

    print("\nSaved results to:")
    print(f"- {RESULTS_DIR / 'baseline_metrics.csv'}")
    print(f"- {RESULTS_DIR / 'baseline_metrics.txt'}")
    print(f"- {RESULTS_DIR / 'baseline_metrics_barplot.png'}")
    print(f"- {PRED_DIR}")


if __name__ == "__main__":
    main()
