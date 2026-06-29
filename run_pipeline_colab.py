import os
import shutil
import subprocess
from pathlib import Path

# ----------------------------
# Settings
# ----------------------------
WORK_DIR = Path("/content/sample_data")
REPO_DIR = WORK_DIR / "Pytorch-UNet"
DATA_ROOT = Path("/content/brain_tumor_unet")

KAGGLE_DATASET = "pkdarabi/brain-tumor-image-dataset-semantic-segmentation"

EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
SCALE = 0.5
CLASSES = 1


def run(command, cwd=None):
    print(f"\nRunning: {command}")
    subprocess.run(command, shell=True, check=True, cwd=cwd)


def clone_repo():
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)

    run("git clone https://github.com/milesial/Pytorch-UNet.git", cwd=WORK_DIR)


def install_requirements():
    run("pip install -r requirements.txt", cwd=REPO_DIR)
    run("pip install kagglehub pycocotools")


def download_dataset():
    import kagglehub

    path = kagglehub.dataset_download(KAGGLE_DATASET)
    print("Dataset downloaded to:", path)
    return Path(path)


def inspect_dataset(dataset_path):
    print("\nDataset structure:")
    for root, dirs, files in os.walk(dataset_path):
        print(root)
        print("Number of files:", len(files))
        print(files[:5])
        print("-" * 50)

    for split in ["train", "valid", "test"]:
        split_path = dataset_path / split
        json_files = [f for f in os.listdir(split_path) if f.endswith(".json")]
        print(split, json_files)


def convert_coco_to_masks(dataset_path):
    import numpy as np
    from PIL import Image
    from pycocotools.coco import COCO

    for split in ["train", "valid", "test"]:
        split_dir = dataset_path / split
        ann_file = split_dir / "_annotations.coco.json"

        img_out = DATA_ROOT / split / "imgs"
        mask_out = DATA_ROOT / split / "masks"

        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        coco = COCO(str(ann_file))

        for img_id in coco.getImgIds():
            img_info = coco.loadImgs(img_id)[0]
            file_name = img_info["file_name"]

            img_path = split_dir / file_name
            image = Image.open(img_path).convert("RGB")
            image.save(img_out / file_name)

            mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)

            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)

            for ann in anns:
                mask = np.maximum(mask, coco.annToMask(ann) * 255)

            mask_name = Path(file_name).stem + ".png"
            Image.fromarray(mask).save(mask_out / mask_name)

    print("\nDone converting COCO annotations to U-Net masks.")


def verify_dataset():
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    img_dir = DATA_ROOT / "train" / "imgs"
    mask_dir = DATA_ROOT / "train" / "masks"

    img_names = sorted([p.stem for p in img_dir.iterdir()])
    mask_names = sorted([p.stem for p in mask_dir.iterdir()])

    print("\nDataset verification")
    print("Images:", len(img_names))
    print("Masks:", len(mask_names))
    print("Matched:", len(set(img_names) & set(mask_names)))
    print("Missing masks:", list(set(img_names) - set(mask_names))[:5])
    print("Missing images:", list(set(mask_names) - set(img_names))[:5])

    name = img_names[0]

    img = Image.open(img_dir / f"{name}.jpg")
    mask = Image.open(mask_dir / f"{name}.png")

    print("Image:", img.size, img.mode)
    print("Mask:", mask.size, mask.mode)
    print("Mask values:", np.unique(np.array(mask)))

    figures_dir = REPO_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("MRI Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Tumor Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(figures_dir / "dataset_example.png", dpi=300, bbox_inches="tight")
    plt.show()


def write_training_files():
    train_py = r'''
import argparse
import csv
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss


dir_img = Path("/content/brain_tumor_unet/train/imgs/")
dir_mask = Path("/content/brain_tumor_unet/train/masks/")
dir_checkpoint = Path("./checkpoints/")
dir_results = Path("./results/")


def train_model(
    model,
    device,
    epochs=5,
    batch_size=4,
    learning_rate=1e-4,
    val_percent=0.1,
    save_checkpoint=True,
    img_scale=0.5,
    amp=False,
    weight_decay=1e-8,
    momentum=0.9,
    gradient_clipping=1.0,
):
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )

    loader_args = dict(
        batch_size=batch_size,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    optimizer = optim.RMSprop(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        foreach=True,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=5,
    )

    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    dir_results.mkdir(exist_ok=True)
    history_file = dir_results / "training_history.csv"

    with open(history_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_dice", "learning_rate"])

    logging.info(f"""
Starting training:
    Epochs:          {epochs}
    Batch size:      {batch_size}
    Learning rate:   {learning_rate}
    Training size:   {n_train}
    Validation size: {n_val}
    Device:          {device.type}
    Image scale:     {img_scale}
    AMP:             {amp}
""")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit="img") as pbar:
            for batch in train_loader:
                images = batch["image"]
                true_masks = batch["mask"]

                images = images.to(
                    device=device,
                    dtype=torch.float32,
                    memory_format=torch.channels_last,
                )

                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(
                    device.type if device.type != "mps" else "cpu",
                    enabled=amp,
                ):
                    masks_pred = model(images)

                    if model.n_classes == 1:
                        bce = criterion(masks_pred.squeeze(1), true_masks.float())
                        dsc = dice_loss(
                            F.sigmoid(masks_pred.squeeze(1)),
                            true_masks.float(),
                            multiclass=False,
                        )
                        loss = bce + dsc
                    else:
                        ce = criterion(masks_pred, true_masks)
                        dsc = dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes)
                            .permute(0, 3, 1, 2)
                            .float(),
                            multiclass=True,
                        )
                        loss = ce + dsc

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                epoch_loss += loss.item()
                pbar.update(images.shape[0])
                pbar.set_postfix(loss=loss.item())

        avg_train_loss = epoch_loss / len(train_loader)

        val_score = evaluate(
            model,
            val_loader,
            device=device,
            amp=amp,
        )

        scheduler.step(val_score)
        lr = optimizer.param_groups[0]["lr"]

        with open(history_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_loss, float(val_score), lr])

        logging.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Dice: {float(val_score):.4f} | "
            f"LR: {lr:.6f}"
        )

        if save_checkpoint:
            dir_checkpoint.mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict["mask_values"] = dataset.mask_values
            torch.save(
                state_dict,
                str(dir_checkpoint / f"checkpoint_epoch{epoch}.pth"),
            )
            logging.info(f"Checkpoint {epoch} saved.")


def get_args():
    parser = argparse.ArgumentParser(description="Train U-Net on brain tumor segmentation dataset")

    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument("--batch-size", "-b", dest="batch_size", type=int, default=4)
    parser.add_argument("--learning-rate", "-l", dest="lr", type=float, default=1e-4)
    parser.add_argument("--load", "-f", type=str, default=False)
    parser.add_argument("--scale", "-s", type=float, default=0.5)
    parser.add_argument("--validation", "-v", dest="val", type=float, default=10.0)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--bilinear", action="store_true", default=False)
    parser.add_argument("--classes", "-c", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    model = UNet(
        n_channels=3,
        n_classes=args.classes,
        bilinear=args.bilinear,
    )

    model = model.to(memory_format=torch.channels_last)
    model.to(device=device)

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        state_dict.pop("mask_values", None)
        model.load_state_dict(state_dict)
        logging.info(f"Model loaded from {args.load}")

    train_model(
        model=model,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        img_scale=args.scale,
        val_percent=args.val / 100,
        amp=args.amp,
    )
'''
    (REPO_DIR / "train.py").write_text(train_py)

    evaluate_and_plot_py = r'''
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet import UNet
from utils.data_loading import BasicDataset
from utils.segmentation_metrics import binary_segmentation_metrics


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

    return {
        key: float(np.mean([m[key] for m in all_metrics]))
        for key in all_metrics[0]
    }


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


def plot_training_history():
    history_path = RESULTS_DIR / "training_history.csv"

    if not history_path.exists():
        print("No training_history.csv found. Skipping training curves.")
        return

    df = pd.read_csv(history_path)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epoch")
    plt.grid(True)
    plt.legend()
    plt.savefig(RESULTS_DIR / "train_loss_vs_epoch.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["val_dice"], marker="o", label="Validation Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Validation Dice vs Epoch")
    plt.grid(True)
    plt.legend()
    plt.savefig(RESULTS_DIR / "val_dice_vs_epoch.png", dpi=300, bbox_inches="tight")
    plt.show()


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
    plot_training_history()
    plot_metrics(metrics)
    save_prediction_examples(model, dataset, device)

    print("\nSaved results to:")
    print(f"- {RESULTS_DIR / 'training_history.csv'}")
    print(f"- {RESULTS_DIR / 'train_loss_vs_epoch.png'}")
    print(f"- {RESULTS_DIR / 'val_dice_vs_epoch.png'}")
    print(f"- {RESULTS_DIR / 'baseline_metrics.csv'}")
    print(f"- {RESULTS_DIR / 'baseline_metrics.txt'}")
    print(f"- {RESULTS_DIR / 'baseline_metrics_barplot.png'}")
    print(f"- {PRED_DIR}")


if __name__ == "__main__":
    main()
'''
    (REPO_DIR / "evaluate_and_plot.py").write_text(evaluate_and_plot_py)

    seg_metrics_py = r'''
import torch


def binary_segmentation_metrics(pred, target, threshold=0.5, eps=1e-7):
    pred = (pred > threshold).float()
    target = target.float()

    pred = pred.view(-1)
    target = target.view(-1)

    tp = ((pred == 1) & (target == 1)).sum().float()
    tn = ((pred == 0) & (target == 0)).sum().float()
    fp = ((pred == 1) & (target == 0)).sum().float()
    fn = ((pred == 0) & (target == 1)).sum().float()

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "accuracy": accuracy.item(),
        "tp": tp.item(),
        "tn": tn.item(),
        "fp": fp.item(),
        "fn": fn.item(),
    }
'''
    (REPO_DIR / "utils" / "segmentation_metrics.py").write_text(seg_metrics_py)


def train_model():
    run(
        f"python train.py --epochs {EPOCHS} "
        f"--batch-size {BATCH_SIZE} "
        f"--learning-rate {LEARNING_RATE} "
        f"--scale {SCALE} "
        f"--classes {CLASSES}",
        cwd=REPO_DIR,
    )


def evaluate_and_plot():
    run("python evaluate_and_plot.py", cwd=REPO_DIR)


def main():
    clone_repo()
    install_requirements()

    dataset_path = download_dataset()
    inspect_dataset(dataset_path)
    convert_coco_to_masks(dataset_path)
    verify_dataset()

    write_training_files()

    train_model()
    evaluate_and_plot()

    print("\nPipeline finished successfully.")
    print("Repository folder:", REPO_DIR)
    print("Results folder:", REPO_DIR / "results")


if __name__ == "__main__":
    main()
