"""
Hybrid CNN-ViT model for brain tumor classification.
Combines CNN backbone, Vision Transformer, and radiomics features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .cnn_backbone import CNNBackbone
from .vit_module import PatchEmbedding, ViTEncoder
from .fusion import FeatureFusion
from .radiomics import LearnableRadiomics


class HybridCNNViT(nn.Module):
    """
    Hybrid CNN-ViT architecture for brain tumor classification.
    
    Architecture:
    1. CNN backbone extracts local texture/shape features
    2. ViT encoder captures global context via self-attention
    3. Optional radiomics branch adds hand-crafted features
    4. Feature fusion combines all sources
    5. Classification head outputs tumor type predictions
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        cnn_backbone: str = 'resnet50',
        cnn_pretrained: bool = True,
        vit_embed_dim: int = 512,
        vit_depth: int = 6,
        vit_num_heads: int = 8,
        vit_mlp_ratio: float = 4.0,
        use_radiomics: bool = True,
        radiomics_dim: int = 128,
        fusion_type: str = 'concat',
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.use_radiomics = use_radiomics
        
        # CNN Backbone
        self.cnn = CNNBackbone(
            backbone_name=cnn_backbone,
            pretrained=cnn_pretrained,
            output_features=True
        )
        cnn_feature_dim = self.cnn.num_features
        
        # Determine feature map size (depends on input size and backbone)
        # For 224x224 input with ResNet: 7x7 feature maps
        self.feature_size = 7
        
        # Patch embedding for ViT
        self.patch_embed = PatchEmbedding(
            feature_size=self.feature_size,
            feature_dim=cnn_feature_dim,
            embed_dim=vit_embed_dim,
            patch_size=1
        )
        
        # ViT Encoder
        self.vit_encoder = ViTEncoder(
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_num_heads,
            mlp_ratio=vit_mlp_ratio,
            dropout=dropout * 0.5
        )
        
        # CNN global pooling for fusion
        self.cnn_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Radiomics branch
        if use_radiomics:
            self.radiomics = LearnableRadiomics(in_channels=3, feature_dim=radiomics_dim)
        else:
            self.radiomics = None
            radiomics_dim = 0
        
        # Feature fusion
        self.fusion = FeatureFusion(
            cnn_dim=cnn_feature_dim,
            vit_dim=vit_embed_dim,
            radiomics_dim=radiomics_dim,
            output_dim=512,
            fusion_type=fusion_type,
            use_radiomics=use_radiomics
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )
        
        # Store attention weights for visualization
        self.attention_weights = None
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input images (B, 3, H, W)
            return_features: Return intermediate features
            return_attention: Return attention weights
            
        Returns:
            Dictionary with 'logits' and optionally 'features', 'attention'
        """
        # CNN backbone
        cnn_features = self.cnn(x)  # (B, C, H, W)
        cnn_pooled = self.cnn_pool(cnn_features)  # (B, C)
        
        # ViT encoder
        patch_embeddings = self.patch_embed(cnn_features)
        vit_output, attention = self.vit_encoder(patch_embeddings, return_attention)
        vit_cls = vit_output[:, 0]  # CLS token
        
        if return_attention:
            self.attention_weights = attention
        
        # Radiomics features
        if self.use_radiomics:
            radiomics_features = self.radiomics(x)
        else:
            radiomics_features = None
        
        # Fusion
        fused = self.fusion(cnn_pooled, vit_cls, radiomics_features)
        
        # Classification
        logits = self.classifier(fused)
        
        output = {'logits': logits}
        
        if return_features:
            output['cnn_features'] = cnn_pooled
            output['vit_features'] = vit_cls
            output['fused_features'] = fused
            if radiomics_features is not None:
                output['radiomics_features'] = radiomics_features
        
        if return_attention:
            output['attention'] = attention
        
        return output
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention maps for visualization."""
        with torch.no_grad():
            output = self.forward(x, return_attention=True)
        return output.get('attention', None)


class BrainTumorClassifier(nn.Module):
    """
    Complete brain tumor classifier with configurable architecture.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        model_config = config.get('model', {})
        
        self.model = HybridCNNViT(
            num_classes=config.get('data', {}).get('num_classes', 4),
            cnn_backbone=model_config.get('cnn_backbone', 'resnet50'),
            cnn_pretrained=model_config.get('cnn_pretrained', True),
            vit_embed_dim=model_config.get('vit_embed_dim', 512),
            vit_depth=model_config.get('vit_depth', 6),
            vit_num_heads=model_config.get('vit_num_heads', 8),
            vit_mlp_ratio=model_config.get('vit_mlp_ratio', 4.0),
            use_radiomics=model_config.get('use_radiomics', True),
            radiomics_dim=model_config.get('radiomics_features', 128),
            fusion_type='concat',
            dropout=model_config.get('dropout', 0.3)
        )
        
        self.num_classes = config.get('data', {}).get('num_classes', 4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        return output['logits']
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get predictions and probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs


def create_model(config: Dict) -> nn.Module:
    """Factory function to create model from config."""
    model_name = config.get('model', {}).get('name', 'hybrid_cnn_vit')
    
    if model_name == 'hybrid_cnn_vit':
        return BrainTumorClassifier(config)
    elif model_name == 'cnn_only':
        return CNNBackbone(
            backbone_name=config['model'].get('cnn_backbone', 'resnet50'),
            pretrained=config['model'].get('cnn_pretrained', True),
            num_classes=config['data'].get('num_classes', 4),
            output_features=False
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
