---
title: Brain Tumor Detection & Classification
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: true
license: mit
tags:
  - medical-imaging
  - brain-tumor
  - classification
  - cnn
  - vision-transformer
  - gradcam
  - pytorch
  - deep-learning
---

# 🧠 Automated Brain Tumor Detection & Classification

A novel deep learning framework for brain tumor classification using a **Hybrid CNN-Vision Transformer** architecture with multimodal fusion, radiomics feature integration, and Grad-CAM explainability.

## 🔬 Model Architecture

```
Input MRI (224×224) → CNN Backbone (ResNet50) → Feature Maps (7×7×2048)
                                                      │
                                        ┌─────────────┼─────────────┐
                                        ▼             ▼             ▼
                                  Global Pool    Patch Embed    Radiomics
                                   (2048-d)     → ViT Encoder   Branch
                                        │        (512-d CLS)   (128-d)
                                        └─────────────┼─────────────┘
                                                      ▼
                                              Feature Fusion
                                                      ▼
                                           Classification Head
                                                      ▼
                                         4-Class Prediction + Grad-CAM
```

### Key Components
- **CNN Backbone**: ResNet50 (ImageNet pretrained) — extracts local texture/shape features
- **Vision Transformer**: 6-layer, 8-head ViT — captures global context and long-range dependencies
- **Learnable Radiomics**: Dual-branch CNN extracting texture and shape features
- **Feature Fusion**: Concatenation-based fusion of CNN (2048-d), ViT (512-d), and Radiomics (128-d) features
- **Explainability**: Grad-CAM heatmaps showing which regions drive predictions

## 📊 Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 99.31% |
| **F1-Score (Weighted)** | 99.30% |
| **ROC-AUC** | 99.92% |
| **5-Fold CV Accuracy** | 98.95% ± 0.27% |

### Ablation Study

| Model | Accuracy | F1-Score | AUC |
|-------|----------|----------|-----|
| CNN Only (ResNet50) | 97.64% | 97.62% | 99.71% |
| ViT Only (ViT-B/16) | 98.43% | 98.43% | 99.94% |
| EfficientNet-B0 | 98.86% | 98.86% | 99.93% |
| **Hybrid CNN-ViT (Ours)** | **99.13%** | **99.13%** | **99.83%** |

## 🎯 Supported Classes

| Class | Description |
|-------|-------------|
| **Glioma** | Most common primary brain tumor, arising from glial cells |
| **Meningioma** | Tumor arising from the meninges surrounding the brain |
| **No Tumor** | Healthy brain MRI with no detectable tumor |
| **Pituitary** | Tumor in the pituitary gland at the base of the brain |

## 📁 Dataset

Trained on the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) — 7,023 images across 4 classes.

## ⚠️ Disclaimer

This is a **research tool** and is **NOT intended for clinical diagnosis**. Always consult qualified medical professionals for medical imaging interpretation.

## 📝 Technical Details

- **Framework**: PyTorch
- **Input Size**: 224×224 RGB
- **Parameters**: ~47M
- **Preprocessing**: CLAHE contrast enhancement + ImageNet normalization
- **Training**: AdamW optimizer, cosine LR scheduler, label smoothing (0.1), mixup augmentation
- **Validation**: 5-fold cross-validation with stratified splits

## 📜 License

MIT License
