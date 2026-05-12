---
license: mit
library_name: pytorch
pipeline_tag: image-classification
tags:
  - brain-tumor
  - medical-imaging
  - cnn
  - vision-transformer
  - hybrid-model
  - mri
  - deep-learning
  - radiomics
  - grad-cam
  - resnet50
datasets:
  - masoudnickparvar/brain-tumor-mri-dataset
metrics:
  - accuracy
  - f1
  - roc_auc
model-index:
  - name: brain-tumor-cnn-vit
    results:
      - task:
          type: image-classification
          name: Brain Tumor Classification
        dataset:
          name: Brain Tumor MRI Dataset
          type: masoudnickparvar/brain-tumor-mri-dataset
        metrics:
          - type: accuracy
            value: 0.98
          - type: f1
            value: 0.97
          - type: roc_auc
            value: 0.99
---

# 🧠 Hybrid CNN-ViT for Brain Tumor Classification

A novel deep learning framework for automated brain tumor detection and classification from MRI images. Combines a **ResNet50 CNN backbone** with a **6-layer Vision Transformer** and **learnable radiomics features** via multimodal fusion.

## Model Description

This model classifies brain MRI scans into **4 categories**:

| Label | Description |
|-------|-------------|
| `glioma` | Glioma tumor |
| `meningioma` | Meningioma tumor |
| `no_tumor` | Healthy brain (no tumor) |
| `pituitary` | Pituitary tumor |

### Architecture

```
Input MRI (224×224×3)
       │
       ├──► ResNet50 CNN ──► Feature Maps (7×7×2048)
       │                          │
       │                    Patch Embedding
       │                          │
       │                    ViT Encoder (6 blocks, 8 heads)
       │                          │
       │                       CLS Token (512-d)
       │
       ├──► Radiomics Branch ──► Texture + Shape Features (128-d)
       │
       └──► CNN Global Pool ──► CNN Features (2048-d)
                                     │
                      ┌──────────────┼──────────────┐
                      │              │              │
                   CNN (2048)    ViT (512)    Radiomics (128)
                      │              │              │
                      └──────── Concat Fusion ──────┘
                                     │
                              MLP Classifier
                                     │
                              4 Class Logits
```

### Key Innovations

1. **Hybrid CNN + ViT**: CNN captures local texture/shape; ViT captures global context and long-range dependencies
2. **Learnable Radiomics**: Dual-branch CNN (texture + shape) providing hand-crafted-style features in a differentiable way
3. **Feature Fusion**: Concatenation-based fusion with LayerNorm and GELU for stable multimodal learning
4. **Self-Supervised Pre-Training**: Masked Autoencoder (MAE) pre-training for better generalization

## Performance

| Model Variant | Accuracy | F1-Score | AUC |
|:---|:---:|:---:|:---:|
| ResNet50 (baseline) | 93% | 0.92 | 0.97 |
| Hybrid CNN-ViT | 96% | 0.95 | 0.99 |
| + Self-Supervised Pre-Training | 97% | 0.96 | 0.99 |
| **+ Radiomics (Full Model)** | **98%** | **0.97** | **0.99** |

## Usage

### Quick Inference

```python
import torch
from PIL import Image
from torchvision import transforms
from model import HybridCNNViT

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridCNNViT(
    num_classes=4,
    cnn_backbone="resnet50",
    cnn_pretrained=False,
    vit_embed_dim=512,
    vit_depth=6,
    vit_num_heads=8,
    use_radiomics=True,
    radiomics_dim=128,
    dropout=0.3,
)

checkpoint = torch.load("best_model.pth", map_location=device)
state_dict = checkpoint.get("model_state_dict", checkpoint)
model.load_state_dict(state_dict, strict=False)
model.eval().to(device)

# Preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

image = Image.open("brain_mri.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output["logits"], dim=-1)
    pred_class = probs.argmax(dim=-1).item()

class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]
print(f"Prediction: {class_names[pred_class]} ({probs[0][pred_class]:.1%})")
```

## Training Details

- **Dataset**: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (~7,000 MRI images)
- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.01)
- **Scheduler**: Cosine annealing with 5-epoch warmup
- **Augmentation**: Random rotation (±15°), horizontal flip, elastic deformation, MixUp (α=0.2)
- **Regularization**: Label smoothing (0.1), gradient clipping (1.0), dropout (0.3)
- **Hardware**: NVIDIA GPU with mixed precision (FP16) training

## Limitations & Ethical Considerations

> ⚠️ **This model is for research and educational purposes only.**

- **Not FDA-approved** for clinical diagnosis
- Trained on a single publicly available dataset — may not generalize to all MRI scanners/protocols
- Should be used as a decision-support tool, not a replacement for radiologist evaluation
- Performance may vary on MRI sequences not seen during training (e.g., contrast-enhanced)

## Citation

```bibtex
@misc{vishnuk2024braintumor,
  title={Hybrid CNN-ViT Framework for Brain Tumor Classification with Radiomics Integration},
  author={Vishnu K},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/ZorroJurro/brain-tumor-cnn-vit}
}
```

## Author

**Vishnu K** — [Hugging Face](https://huggingface.co/ZorroJurro) · [GitHub](https://github.com/ZorroJurro)
