import torch


def binary_segmentation_metrics(pred, target, threshold=0.5, eps=1e-7):
    """
    pred: model output probabilities or binary mask
    target: ground truth mask with values 0/1
    """

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
