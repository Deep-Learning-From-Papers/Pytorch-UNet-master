# U-Net From Paper

> A research-oriented implementation and study of the original U-Net architecture for medical image segmentation.

---

# U-Net Architecture

<p align="center">
  <img src="figures/unet_architecture.png" width="850">
</p>

**Original U-Net architecture proposed by Ronneberger et al. (MICCAI 2015).**

The network consists of:

- Contracting (Encoder) path
- Bottleneck
- Expanding (Decoder) path
- Skip Connections
- Pixel-wise segmentation output

This repository explains every component of the architecture and reproduces the original implementation before performing systematic experiments and improvements.

---

# Current Status

## Project Stage

🟢 Dataset Prepared

Current progress

- ✅ Repository created
- ✅ Original U-Net repository imported
- ✅ Brain Tumor dataset downloaded
- ✅ COCO annotations converted into binary masks
- ✅ Image–mask pairs verified
- ✅ Dataset visualization completed
- ⏳ Training preparation
- ⏳ Model training
- ⏳ Evaluation
- ⏳ Hyperparameter experiments

---

# Dataset

Dataset

**Brain Tumor Image Dataset – Semantic Segmentation**

Task

Binary Semantic Segmentation

Classes

- Background
- Tumor

Original dataset

- 2146 MRI images
- COCO segmentation annotations

Current dataset structure

```text
brain_tumor_unet/

    train/
        imgs/
        masks/

    valid/
        imgs/
        masks/

    test/
        imgs/
        masks/
```

Image size

```
640 × 640
```

Mask values

```
0   Background

255 Tumor
```

---

# Planned Experiments

## Hyperparameters

- Learning Rate
- Batch Size
- Weight Decay
- Epochs
- Momentum

## Optimizers

- Adam
- AdamW
- SGD
- RMSprop

## Loss Functions

- Cross Entropy
- Dice Loss
- BCE + Dice
- Focal Loss
- Tversky Loss

## Evaluation Metrics

- Dice
- IoU
- Precision
- Recall
- F1-score
- Accuracy
- Confusion Matrix

## Architectures

- Original U-Net
- Attention U-Net
- UNet++
- ResUNet
- UNETR
- SwinUNETR

---

# Repository Structure

```text
paper/
implementation/
dataset/
experiments/
results/
figures/
```

---

# Roadmap

- [x] Study original paper
- [x] Understand U-Net architecture
- [x] Prepare dataset
- [x] Convert COCO annotations
- [ ] Train baseline U-Net
- [ ] Evaluate baseline
- [ ] Hyperparameter tuning
- [ ] Optimizer comparison
- [ ] Loss function comparison
- [ ] Architecture comparison

---

# Acknowledgment

This project started from the excellent open-source implementation:

https://github.com/milesial/Pytorch-UNet

The goal of this repository is educational and research-oriented. Throughout the project, the original implementation will be analyzed, modified, and gradually replaced while documenting every experiment and design decision.
