# U-Net From Paper

> A research-oriented implementation and study of the original U-Net architecture for medical image segmentation.

---

# Current Status

## Project Stage

🟢 Dataset Prepared

Current progress:

- ✅ Repository created
- ✅ Original U-Net repository imported
- ✅ Brain Tumor dataset downloaded
- ✅ COCO annotations converted into binary segmentation masks
- ✅ Image–mask pairs verified
- ✅ Dataset visualization completed
- ⏳ Training preparation
- ⏳ Model training
- ⏳ Evaluation
- ⏳ Hyperparameter experiments

---

# Dataset

Dataset:

**Brain Tumor Image Dataset – Semantic Segmentation**

Task

Binary Semantic Segmentation

Classes

- Background
- Tumor

Original dataset contains

- 2146 MRI images
- COCO segmentation annotations

The annotations were converted into binary PNG masks using `pycocotools`.

Current dataset structure

```
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
0 = Background

255 = Tumor
```

---

# Dataset Preparation Pipeline

```
COCO Dataset

↓

Read Annotation JSON

↓

Convert Polygon

↓

Binary Mask (.png)

↓

Verify Image–Mask Pairs

↓

Ready for U-Net Training
```

---

# Repository Goals

This repository is **not only an implementation** of U-Net.

The objective is to understand every component of the original paper and investigate how different design choices influence segmentation performance.

The project includes

- Understanding the original paper
- Understanding every line of the PyTorch implementation
- Reproducing the original training pipeline
- Training on a medical dataset
- Performing systematic experiments
- Documenting all observations

---

# Planned Experiments

The following experiments will be performed.

## Training Hyperparameters

- Learning Rate
- Batch Size
- Number of Epochs
- Weight Decay
- Momentum

---

## Optimizers

- Adam
- AdamW
- RMSprop
- SGD

---

## Loss Functions

- Cross Entropy
- Dice Loss
- BCE + Dice Loss
- Focal Loss
- Tversky Loss

---

## Data Augmentation

- Horizontal Flip
- Rotation
- Random Crop
- Brightness
- Elastic Transformation

---

## Evaluation Metrics

The trained model will be evaluated using

- Dice Score
- Dice Loss
- IoU (Intersection over Union)
- Precision
- Recall
- F1 Score
- Accuracy
- Confusion Matrix

---

## Model Variants

After reproducing the original U-Net, the following architectures will be investigated.

- Attention U-Net
- UNet++
- ResUNet
- UNETR
- SwinUNETR

---

# Current Repository Structure

```
paper/
implementation/
dataset/
experiments/
results/
figures/
```

---

# Project Roadmap

- [x] Study original U-Net paper
- [x] Understand repository structure
- [x] Convert COCO annotations
- [x] Verify dataset
- [ ] Train original U-Net
- [ ] Evaluate baseline performance
- [ ] Compare hyperparameters
- [ ] Compare optimizers
- [ ] Compare loss functions
- [ ] Compare model variants
- [ ] Final benchmark

---

# Acknowledgment

This project started from the open-source implementation:

https://github.com/milesial/Pytorch-UNet

The purpose of this repository is educational and research-oriented.
The implementation will gradually evolve through independent modifications,
experiments, and reimplementation while documenting the complete learning process.
