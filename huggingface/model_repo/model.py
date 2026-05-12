"""
Hybrid CNN-ViT Model for Brain Tumor Classification.

Self-contained module bundling CNN backbone, Vision Transformer,
Radiomics, and Feature Fusion into a single file for Hugging Face deployment.

Architecture:
    1. ResNet50 CNN backbone → local texture/shape features
    2. Vision Transformer encoder → global context via self-attention
    3. Learnable Radiomics branch → texture + shape features
    4. Feature Fusion → concatenation + MLP projection
    5. Classification Head → 4-class tumor prediction

Author: Vishnu K (ZorroJurro)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Optional, Tuple
from einops import rearrange, repeat


# =============================================================================
# 1. CNN Backbone
# =============================================================================

class CNNBackbone(nn.Module):
    """ResNet50 backbone for local feature extraction from brain MRI."""

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        output_features: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.output_features = output_features

        resnet_configs = {
            "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, 512),
            "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, 512),
            "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, 2048),
            "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V2, 2048),
        }

        if self.backbone_name not in resnet_configs:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        model_fn, weights, self.num_features = resnet_configs[self.backbone_name]
        model = model_fn(weights=weights if pretrained else None)

        # Remove final avg pool and fc to get feature maps
        self.backbone = nn.Sequential(*list(model.children())[:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# =============================================================================
# 2. Vision Transformer Components
# =============================================================================

class PatchEmbedding(nn.Module):
    """Convert CNN feature maps to patch embeddings for ViT."""

    def __init__(
        self,
        feature_size: int = 7,
        feature_dim: int = 2048,
        embed_dim: int = 512,
        patch_size: int = 1,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.patch_size = patch_size
        self.num_patches = (feature_size // patch_size) ** 2

        if patch_size == 1:
            self.projection = nn.Linear(feature_dim, embed_dim)
        else:
            self.projection = nn.Conv2d(
                feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size
            )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        if self.patch_size == 1:
            x = rearrange(x, "b c h w -> b (h w) c")
            x = self.projection(x)
        else:
            x = self.projection(x)
            x = rearrange(x, "b c h w -> b (h w) c")

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=B)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding[:, : x.size(1)]
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention for Vision Transformer."""

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        self.attention_weights = attn.detach()

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_dropout(x)

        if return_attention:
            return x, attn
        return x, None


class TransformerBlock(nn.Module):
    """Transformer encoder block: MHSA + FFN with residual connections."""

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim, num_heads, dropout, attention_dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.attention_weights = None

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn = self.attn(self.norm1(x), return_attention)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        self.attention_weights = attn
        if return_attention:
            return x, attn
        return x, None


class ViTEncoder(nn.Module):
    """Vision Transformer encoder: stack of TransformerBlocks."""

    def __init__(
        self,
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim, num_heads, mlp_ratio, dropout, attention_dropout
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.attention_weights_all = []

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        self.attention_weights_all = []
        for block in self.blocks:
            x, attn = block(x, return_attention)
            if return_attention and attn is not None:
                self.attention_weights_all.append(attn)
        x = self.norm(x)
        if return_attention:
            return x, self.attention_weights_all
        return x, None


# =============================================================================
# 3. Learnable Radiomics
# =============================================================================

class LearnableRadiomics(nn.Module):
    """CNN-based radiomics: texture + shape branches fused together."""

    def __init__(self, in_channels: int = 3, feature_dim: int = 128):
        super().__init__()
        self.texture_branch = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, feature_dim // 2),
        )
        self.shape_branch = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, feature_dim // 2),
        )
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        texture = self.texture_branch(x)
        shape = self.shape_branch(x)
        combined = torch.cat([texture, shape], dim=-1)
        return self.fusion(combined)


# =============================================================================
# 4. Feature Fusion
# =============================================================================

class FeatureFusion(nn.Module):
    """Fuse CNN, ViT, and radiomics features via concatenation + MLP."""

    def __init__(
        self,
        cnn_dim: int = 2048,
        vit_dim: int = 512,
        radiomics_dim: int = 128,
        output_dim: int = 512,
        fusion_type: str = "concat",
        use_radiomics: bool = True,
    ):
        super().__init__()
        self.use_radiomics = use_radiomics
        self.fusion_type = fusion_type

        total_dim = cnn_dim + vit_dim + (radiomics_dim if use_radiomics else 0)

        if fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(total_dim, output_dim * 2),
                nn.LayerNorm(output_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim * 2, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
            )
        self.output_dim = output_dim

    def forward(
        self,
        cnn_features: torch.Tensor,
        vit_features: torch.Tensor,
        radiomics_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_radiomics and radiomics_features is not None:
            x = torch.cat([cnn_features, vit_features, radiomics_features], dim=-1)
        else:
            x = torch.cat([cnn_features, vit_features], dim=-1)
        return self.fusion(x)


# =============================================================================
# 5. Complete Hybrid CNN-ViT Model
# =============================================================================

class HybridCNNViT(nn.Module):
    """
    Hybrid CNN-ViT for Brain Tumor Classification.

    Pipeline:
        Image → CNN Backbone → Feature Maps
                                  ↓
                         Patch Embedding → ViT Encoder → CLS Token
                                  ↓
        Image → Radiomics Branch → Radiomics Features
                                  ↓
                    [CNN Pooled | ViT CLS | Radiomics] → Fusion → Classifier
    """

    def __init__(
        self,
        num_classes: int = 4,
        cnn_backbone: str = "resnet50",
        cnn_pretrained: bool = True,
        vit_embed_dim: int = 512,
        vit_depth: int = 6,
        vit_num_heads: int = 8,
        vit_mlp_ratio: float = 4.0,
        use_radiomics: bool = True,
        radiomics_dim: int = 128,
        fusion_type: str = "concat",
        dropout: float = 0.3,
    ):
        super().__init__()
        self.use_radiomics = use_radiomics

        # CNN Backbone
        self.cnn = CNNBackbone(
            backbone_name=cnn_backbone,
            pretrained=cnn_pretrained,
            output_features=True,
        )
        cnn_feature_dim = self.cnn.num_features
        self.feature_size = 7

        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            feature_size=self.feature_size,
            feature_dim=cnn_feature_dim,
            embed_dim=vit_embed_dim,
            patch_size=1,
        )

        # ViT Encoder
        self.vit_encoder = ViTEncoder(
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_num_heads,
            mlp_ratio=vit_mlp_ratio,
            dropout=dropout * 0.5,
        )

        # CNN global pooling
        self.cnn_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())

        # Radiomics branch
        if use_radiomics:
            self.radiomics = LearnableRadiomics(in_channels=3, feature_dim=radiomics_dim)
        else:
            self.radiomics = None
            radiomics_dim = 0

        # Fusion
        self.fusion = FeatureFusion(
            cnn_dim=cnn_feature_dim,
            vit_dim=vit_embed_dim,
            radiomics_dim=radiomics_dim,
            output_dim=512,
            fusion_type=fusion_type,
            use_radiomics=use_radiomics,
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

        self.attention_weights = None

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # CNN backbone
        cnn_features = self.cnn(x)
        cnn_pooled = self.cnn_pool(cnn_features)

        # ViT encoder
        patch_embeddings = self.patch_embed(cnn_features)
        vit_output, attention = self.vit_encoder(patch_embeddings, return_attention)
        vit_cls = vit_output[:, 0]

        if return_attention:
            self.attention_weights = attention

        # Radiomics
        if self.use_radiomics:
            radiomics_features = self.radiomics(x)
        else:
            radiomics_features = None

        # Fusion + Classification
        fused = self.fusion(cnn_pooled, vit_cls, radiomics_features)
        logits = self.classifier(fused)

        output = {"logits": logits}
        if return_features:
            output["cnn_features"] = cnn_pooled
            output["vit_features"] = vit_cls
            output["fused_features"] = fused
            if radiomics_features is not None:
                output["radiomics_features"] = radiomics_features
        if return_attention:
            output["attention"] = attention

        return output


class BrainTumorClassifier(nn.Module):
    """Top-level wrapper that creates HybridCNNViT from config dict."""

    def __init__(self, config: Dict):
        super().__init__()
        model_config = config.get("model", {})
        self.model = HybridCNNViT(
            num_classes=config.get("data", {}).get("num_classes", 4),
            cnn_backbone=model_config.get("cnn_backbone", "resnet50"),
            cnn_pretrained=model_config.get("cnn_pretrained", True),
            vit_embed_dim=model_config.get("vit_embed_dim", 512),
            vit_depth=model_config.get("vit_depth", 6),
            vit_num_heads=model_config.get("vit_num_heads", 8),
            vit_mlp_ratio=model_config.get("vit_mlp_ratio", 4.0),
            use_radiomics=model_config.get("use_radiomics", True),
            radiomics_dim=model_config.get("radiomics_features", 128),
            fusion_type="concat",
            dropout=model_config.get("dropout", 0.3),
        )
        self.num_classes = config.get("data", {}).get("num_classes", 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        return output["logits"]

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs
