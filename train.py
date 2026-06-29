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
