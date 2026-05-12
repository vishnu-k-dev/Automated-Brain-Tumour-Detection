"""
Brain Tumor Detection — Gradio Space
Hybrid CNN-ViT model with Grad-CAM explainability.

Author: Vishnu K (ZorroJurro)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import cv2
import gradio as gr
from huggingface_hub import hf_hub_download
from einops import rearrange, repeat
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Model Architecture (self-contained)
# =============================================================================

class CNNBackbone(nn.Module):
    def __init__(self, backbone_name="resnet50", pretrained=False, output_features=True):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.output_features = output_features
        configs = {
            "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, 2048),
        }
        model_fn, weights, self.num_features = configs[self.backbone_name]
        model = model_fn(weights=weights if pretrained else None)
        self.backbone = nn.Sequential(*list(model.children())[:-2])

    def forward(self, x):
        return self.backbone(x)


class PatchEmbedding(nn.Module):
    def __init__(self, feature_size=7, feature_dim=2048, embed_dim=512, patch_size=1):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (feature_size // patch_size) ** 2
        self.projection = nn.Linear(feature_dim, embed_dim) if patch_size == 1 else nn.Conv2d(feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)

    def forward(self, x):
        B = x.shape[0]
        if self.patch_size == 1:
            x = rearrange(x, "b c h w -> b (h w) c")
            x = self.projection(x)
        else:
            x = self.projection(x)
            x = rearrange(x, "b c h w -> b (h w) c")
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=B)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding[:, :x.size(1)]
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1, attention_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self, x, return_attention=False):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        self.attention_weights = attn.detach()
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj_dropout(self.proj(x))
        return (x, attn) if return_attention else (x, None)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, mlp_ratio=4.0, dropout=0.1, attention_dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout, attention_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, mlp_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_hidden, embed_dim), nn.Dropout(dropout))
        self.attention_weights = None

    def forward(self, x, return_attention=False):
        attn_out, attn = self.attn(self.norm1(x), return_attention)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        self.attention_weights = attn
        return (x, attn) if return_attention else (x, None)


class ViTEncoder(nn.Module):
    def __init__(self, embed_dim=512, depth=6, num_heads=8, mlp_ratio=4.0, dropout=0.1, attention_dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, attention_dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.attention_weights_all = []

    def forward(self, x, return_attention=False):
        self.attention_weights_all = []
        for block in self.blocks:
            x, attn = block(x, return_attention)
            if return_attention and attn is not None:
                self.attention_weights_all.append(attn)
        x = self.norm(x)
        return (x, self.attention_weights_all) if return_attention else (x, None)


class LearnableRadiomics(nn.Module):
    def __init__(self, in_channels=3, feature_dim=128):
        super().__init__()
        self.texture_branch = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, feature_dim // 2))
        self.shape_branch = nn.Sequential(nn.Conv2d(in_channels, 32, 5, padding=2), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 5, padding=2), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, feature_dim // 2))
        self.fusion = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.LayerNorm(feature_dim), nn.ReLU())

    def forward(self, x):
        return self.fusion(torch.cat([self.texture_branch(x), self.shape_branch(x)], dim=-1))


class FeatureFusion(nn.Module):
    def __init__(self, cnn_dim=2048, vit_dim=512, radiomics_dim=128, output_dim=512, use_radiomics=True):
        super().__init__()
        self.use_radiomics = use_radiomics
        total_dim = cnn_dim + vit_dim + (radiomics_dim if use_radiomics else 0)
        self.fusion = nn.Sequential(nn.Linear(total_dim, output_dim * 2), nn.LayerNorm(output_dim * 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(output_dim * 2, output_dim), nn.LayerNorm(output_dim), nn.GELU())

    def forward(self, cnn_features, vit_features, radiomics_features=None):
        parts = [cnn_features, vit_features]
        if self.use_radiomics and radiomics_features is not None:
            parts.append(radiomics_features)
        return self.fusion(torch.cat(parts, dim=-1))


class HybridCNNViT(nn.Module):
    def __init__(self, num_classes=4, cnn_backbone="resnet50", cnn_pretrained=False,
                 vit_embed_dim=512, vit_depth=6, vit_num_heads=8, vit_mlp_ratio=4.0,
                 use_radiomics=True, radiomics_dim=128, fusion_type="concat", dropout=0.3):
        super().__init__()
        self.use_radiomics = use_radiomics
        self.cnn = CNNBackbone(backbone_name=cnn_backbone, pretrained=cnn_pretrained, output_features=True)
        cnn_feature_dim = self.cnn.num_features
        self.patch_embed = PatchEmbedding(feature_size=7, feature_dim=cnn_feature_dim, embed_dim=vit_embed_dim, patch_size=1)
        self.vit_encoder = ViTEncoder(embed_dim=vit_embed_dim, depth=vit_depth, num_heads=vit_num_heads, mlp_ratio=vit_mlp_ratio, dropout=dropout * 0.5)
        self.cnn_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.radiomics = LearnableRadiomics(in_channels=3, feature_dim=radiomics_dim) if use_radiomics else None
        if not use_radiomics:
            radiomics_dim = 0
        self.fusion = FeatureFusion(cnn_dim=cnn_feature_dim, vit_dim=vit_embed_dim, radiomics_dim=radiomics_dim, output_dim=512, use_radiomics=use_radiomics)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout * 0.5), nn.Linear(256, num_classes))
        self.attention_weights = None

    def forward(self, x, return_features=False, return_attention=False):
        cnn_features = self.cnn(x)
        cnn_pooled = self.cnn_pool(cnn_features)
        patch_embeddings = self.patch_embed(cnn_features)
        vit_output, attention = self.vit_encoder(patch_embeddings, return_attention)
        vit_cls = vit_output[:, 0]
        if return_attention:
            self.attention_weights = attention
        radiomics_features = self.radiomics(x) if self.use_radiomics else None
        fused = self.fusion(cnn_pooled, vit_cls, radiomics_features)
        logits = self.classifier(fused)
        output = {"logits": logits}
        if return_features:
            output["cnn_features"] = cnn_pooled
            output["vit_features"] = vit_cls
            output["fused_features"] = fused
        if return_attention:
            output["attention"] = attention
        return output


class BrainTumorClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        mc = config.get("model", {})
        self.model = HybridCNNViT(
            num_classes=config.get("data", {}).get("num_classes", 4),
            cnn_backbone=mc.get("cnn_backbone", "resnet50"),
            cnn_pretrained=mc.get("cnn_pretrained", False),
            vit_embed_dim=mc.get("vit_embed_dim", 512),
            vit_depth=mc.get("vit_depth", 6),
            vit_num_heads=mc.get("vit_num_heads", 8),
            vit_mlp_ratio=mc.get("vit_mlp_ratio", 4.0),
            use_radiomics=mc.get("use_radiomics", True),
            radiomics_dim=mc.get("radiomics_features", 128),
            dropout=mc.get("dropout", 0.3),
        )
        self.num_classes = config.get("data", {}).get("num_classes", 4)

    def forward(self, x):
        return self.model(x)["logits"]


# =============================================================================
# Grad-CAM Implementation
# =============================================================================

class GradCAM:
    """Simplified Grad-CAM for the CNN backbone."""

    def __init__(self, model: HybridCNNViT):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        # Hook into the last conv layer of the CNN backbone
        target_layer = self.model.cnn.backbone[-1]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        self.model.eval()
        input_tensor.requires_grad_(True)

        output = self.model(input_tensor)
        logits = output["logits"]

        if target_class is None:
            target_class = logits.argmax(dim=-1).item()

        self.model.zero_grad()
        logits[0, target_class].backward()

        gradients = self.gradients
        activations = self.activations

        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam


def create_gradcam_overlay(
    original_image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Create a Grad-CAM heatmap overlay on the original image."""
    h, w = original_image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = np.float32(heatmap) * alpha + np.float32(original_image) * (1 - alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay


# =============================================================================
# Model Loading
# =============================================================================

REPO_ID = "Zorrojurro/brain-tumor-cnn-vit"
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CLASS_EMOJIS = {"Glioma": "🔴", "Meningioma": "🟠", "No Tumor": "🟢", "Pituitary": "🟡"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model():
    """Download and load the model from Hugging Face Hub."""
    print("📥 Downloading model from Hugging Face Hub...")

    # Download checkpoint
    checkpoint_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="best_model.pth",
        cache_dir="./model_cache",
    )

    # Create model
    config = {
        "data": {"num_classes": 4},
        "model": {
            "cnn_backbone": "resnet50",
            "cnn_pretrained": False,
            "vit_embed_dim": 512,
            "vit_depth": 6,
            "vit_num_heads": 8,
            "vit_mlp_ratio": 4.0,
            "use_radiomics": True,
            "radiomics_features": 128,
            "dropout": 0.3,
        },
    }

    classifier = BrainTumorClassifier(config)
    model = classifier.model

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Handle key prefix mismatches
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove 'model.' prefix if present
        new_key = k.replace("model.", "") if k.startswith("model.") else k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval().to(DEVICE)

    print(f"✅ Model loaded on {DEVICE}")
    return model


# Load model at startup
MODEL = load_model()
GRADCAM = GradCAM(MODEL)


# =============================================================================
# Prediction Function
# =============================================================================

def predict(image: Image.Image):
    """Run prediction and generate Grad-CAM visualization."""
    if image is None:
        return None, None, "Please upload an image."

    # Convert to RGB
    image = image.convert("RGB")
    original_np = np.array(image)

    # Preprocess
    input_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    # Forward pass with gradients for Grad-CAM
    with torch.enable_grad():
        cam = GRADCAM.generate(input_tensor)

    # Get predictions
    with torch.no_grad():
        output = MODEL(input_tensor)
        logits = output["logits"]
        probs = F.softmax(logits, dim=-1)[0]

    # Build confidence dict
    confidences = {}
    for i, name in enumerate(CLASS_NAMES):
        emoji = CLASS_EMOJIS[name]
        confidences[f"{emoji} {name}"] = float(probs[i])

    # Grad-CAM overlay
    gradcam_overlay = create_gradcam_overlay(original_np, cam, alpha=0.45)

    # Predicted class info
    pred_idx = probs.argmax().item()
    pred_name = CLASS_NAMES[pred_idx]
    pred_conf = probs[pred_idx].item()
    emoji = CLASS_EMOJIS[pred_name]

    summary = f"## {emoji} {pred_name}\n**Confidence:** {pred_conf:.1%}\n\n"
    if pred_name == "No Tumor":
        summary += "✅ No tumor detected in the MRI scan."
    else:
        summary += f"⚠️ Potential **{pred_name.lower()}** detected. Please consult a medical professional."

    return confidences, gradcam_overlay, summary


# =============================================================================
# Gradio UI
# =============================================================================

CUSTOM_CSS = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}
.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
}
.gr-button-primary:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}
footer {visibility: hidden}
"""

DESCRIPTION = """
# 🧠 Brain Tumor Detection — Hybrid CNN-ViT

Upload a brain MRI scan for instant AI-powered classification with **Grad-CAM explainability**.

**Model Architecture**: ResNet50 (CNN) + 6-Layer Vision Transformer + Learnable Radiomics  
**Classes**: Glioma · Meningioma · No Tumor · Pituitary  
**Performance**: 98% Accuracy · 0.97 F1-Score · 0.99 AUC

> ⚠️ *For research and educational purposes only. Not a substitute for professional medical diagnosis.*
"""

with gr.Blocks(
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
    ),
    title="Brain Tumor Detection — CNN-ViT",
) as demo:

    gr.Markdown(DESCRIPTION)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil",
                label="Upload Brain MRI",
                height=350,
            )
            predict_btn = gr.Button(
                "🔬 Analyze MRI",
                variant="primary",
                size="lg",
            )

        with gr.Column(scale=1):
            gradcam_output = gr.Image(
                label="Grad-CAM Visualization",
                height=350,
            )

    with gr.Row():
        with gr.Column(scale=1):
            label_output = gr.Label(
                label="Classification Confidence",
                num_top_classes=4,
            )
        with gr.Column(scale=1):
            summary_output = gr.Markdown(
                label="Diagnosis Summary",
            )

    predict_btn.click(
        fn=predict,
        inputs=[input_image],
        outputs=[label_output, gradcam_output, summary_output],
    )

    gr.Markdown(
        """
        ---
        **Built by [Vishnu K](https://huggingface.co/ZorroJurro)** · 
        [Model Card](https://huggingface.co/ZorroJurro/brain-tumor-cnn-vit) · 
        [GitHub](https://github.com/ZorroJurro)
        """
    )


if __name__ == "__main__":
    demo.launch()
