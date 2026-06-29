# Baseline U-Net for Brain Tumor Segmentation

## Overview

This project presents a baseline implementation of the U-Net architecture for binary brain tumor segmentation using PyTorch. The objective was to build a complete semantic segmentation pipeline, including dataset preparation, model training, quantitative evaluation, and qualitative analysis.

The implementation includes:

* Dataset preprocessing
* COCO annotation conversion to segmentation masks
* U-Net training
* Model evaluation
* Quantitative performance metrics
* Prediction visualization
* Dataset quality analysis

---

# Dataset

The experiments were conducted using the **Brain Tumor Image Dataset: Semantic Segmentation**, downloaded from Kaggle.

Dataset characteristics:

* 2,146 MRI images
* Binary segmentation
* Class 0: Background
* Class 1: Tumor
* Original annotations provided in COCO format

The COCO annotations were converted into binary PNG masks before training.

---

# Model

The segmentation model is based on the original U-Net architecture.

Configuration:

* Framework: PyTorch
* Input size: 320 × 320
* Input channels: 3
* Output channels: 1
* Loss function:

  * Binary Cross Entropy with Logits Loss
  * Dice Loss
* Optimizer:

  * RMSProp
* Learning rate:

  * 1 × 10⁻⁴
* Batch size:

  * 4
* Number of epochs:

  * 10

---

# Training Pipeline

The implemented training pipeline includes:

* automatic dataset loading
* training / validation split
* checkpoint saving
* validation after each epoch
* Dice score computation
* quantitative segmentation metrics
* prediction visualization

The evaluation metrics include:

* Dice coefficient
* Intersection over Union (IoU)
* Precision
* Recall
* F1 Score
* Accuracy
* True Positive
* False Positive
* True Negative
* False Negative

---

# Validation Results

The final validation performance after training was:

| Metric    |  Score |
| --------- | -----: |
| Dice      | 0.4897 |
| IoU       | 0.3401 |
| Precision | 0.5924 |
| Recall    | 0.4716 |
| F1 Score  | 0.4986 |
| Accuracy  | 0.9678 |

Confusion statistics:

| Metric |  Value |
| ------ | -----: |
| TP     |   6585 |
| TN     | 387206 |
| FP     |   4640 |
| FN     |   8324 |

---

# Qualitative Results

Example predictions are provided in the `results/predictions` directory.

The model successfully identifies the approximate tumor location but fails to recover accurate tumor boundaries.

Example outputs include:

* MRI image
* Ground-truth mask
* Predicted segmentation

---

# Dataset Investigation

During qualitative evaluation, an unexpected behavior was observed.

Although the network converged during training, the predicted masks appeared smooth and approximately elliptical rather than matching anatomical tumor boundaries.

To investigate this issue, the original COCO annotations were visualized.

The analysis revealed that the dataset annotations do **not** represent precise tumor contours.

Instead, the annotations correspond to coarse rectangular regions surrounding the tumor.

The converted binary masks therefore inherit this rectangular structure.

Consequently, the model learns to segment these coarse regions rather than the true tumor shape.

This behavior is illustrated by comparing:

* original MRI
* original COCO annotation
* generated binary mask
* predicted segmentation

---

# Discussion

The obtained Dice score of approximately **0.49** indicates that the implemented U-Net successfully learns the annotation distribution contained in the dataset.

However, qualitative inspection demonstrates that the dataset itself limits the achievable segmentation performance.

The implementation of the model, training procedure, and evaluation pipeline were verified to function correctly.

The primary limitation originates from the annotation quality rather than the segmentation architecture.

Because the annotations represent rectangular regions instead of expert-defined tumor contours, the model cannot learn anatomically accurate tumor boundaries.

---

# Limitations

The current dataset presents several limitations:

* coarse rectangular annotations
* lack of precise pixel-level tumor boundaries
* simplified binary labels
* not representative of clinical segmentation datasets

These limitations restrict the maximum achievable segmentation accuracy.

---

# Future Work

Future improvements include:

* evaluation on the BraTS 2021 dataset
* training with expert neuroradiologist annotations
* multi-class tumor segmentation
* comparison with Attention U-Net and UNet++
* implementation of 3D U-Net for volumetric MRI segmentation

---

# Conclusion

This project demonstrates a complete baseline implementation of U-Net for brain tumor segmentation using PyTorch.

The training pipeline, evaluation framework, and visualization tools were successfully implemented.

Although the quantitative results demonstrate that the network learns the provided annotations, qualitative analysis reveals that the dataset contains coarse rectangular masks instead of accurate tumor contours.

Therefore, the reported performance reflects both the capabilities of the baseline U-Net and the limitations of the selected dataset.

Future work will evaluate the same implementation on expert-annotated datasets such as BraTS 2021, which provide accurate voxel-level tumor segmentations and are widely accepted as the benchmark for brain tumor segmentation research.
