import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import dice_coeff, multiclass_dice_coeff
from utils.segmentation_metrics import binary_segmentation_metrics


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)

    dice_score = 0
    all_metrics = []

    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc="Validation round", unit="batch", leave=False):
            image = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = batch["mask"].to(device=device, dtype=torch.long)

            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, \
                    "True mask indices should be in [0, 1]"

                mask_prob = torch.sigmoid(mask_pred)
                mask_bin = (mask_prob > 0.5).float()

                dice_score += dice_coeff(mask_bin.squeeze(1), mask_true.float(), reduce_batch_first=False)

                metrics = binary_segmentation_metrics(
                    pred=mask_prob.squeeze(1),
                    target=mask_true.float(),
                    threshold=0.5
                )
                all_metrics.append(metrics)

            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, \
                    "True mask indices should be in [0, n_classes["

                mask_true_onehot = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred_onehot = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()

                dice_score += multiclass_dice_coeff(
                    mask_pred_onehot[:, 1:],
                    mask_true_onehot[:, 1:],
                    reduce_batch_first=False
                )

    net.train()

    avg_dice = dice_score / max(num_val_batches, 1)

    if all_metrics:
        avg_metrics = {
            key: sum(m[key] for m in all_metrics) / len(all_metrics)
            for key in all_metrics[0]
        }

        print("\nValidation Metrics")
        print("------------------")
        for key, value in avg_metrics.items():
            print(f"{key}: {value:.4f}")

    return avg_dice, avg_metrics
