"""
CNN Backbone for feature extraction.
Extracts local texture and shape features from brain MRI images.
Supports multiple backbone architectures: ResNet, EfficientNet, etc.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Optional, Tuple
import timm


class CNNBackbone(nn.Module):
    """
    CNN backbone for extracting local features from brain MRI images.
    
    Returns feature maps (not flattened) for downstream processing by
    Vision Transformer or classification head.
    
    Supported backbones:
    - ResNet18, ResNet34, ResNet50, ResNet101
    - EfficientNet-B0 through B7
    - ConvNeXt-Tiny, Small, Base
    """
    
    def __init__(
        self,
        backbone_name: str = 'resnet50',
        pretrained: bool = True,
        num_classes: int = 4,
        freeze_layers: int = 0,
        output_features: bool = True,
        feature_dim: Optional[int] = None
    ):
        """
        Args:
            backbone_name: Name of the backbone architecture
            pretrained: Whether to use ImageNet pretrained weights
            num_classes: Number of output classes
            freeze_layers: Number of initial layers to freeze
            output_features: If True, output feature maps; else output classification logits
            feature_dim: Target feature dimension (optional projection)
        """
        super().__init__()
        
        self.backbone_name = backbone_name.lower()
        self.output_features = output_features
        self.feature_dim = feature_dim
        
        # Initialize backbone
        self.backbone, self.num_features = self._create_backbone(pretrained)
        
        # Freeze specified layers
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
        
        # Optional feature projection
        if feature_dim is not None:
            self.projection = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.num_features, feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            )
        else:
            self.projection = None
        
        # Classification head (if not outputting features)
        if not output_features:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(self.num_features, num_classes)
            )
        else:
            self.classifier = None
    
    def _create_backbone(self, pretrained: bool) -> Tuple[nn.Module, int]:
        """Create the backbone network."""
        
        if self.backbone_name.startswith('resnet'):
            return self._create_resnet(pretrained)
        elif self.backbone_name.startswith('efficientnet'):
            return self._create_efficientnet(pretrained)
        elif self.backbone_name.startswith('convnext'):
            return self._create_convnext(pretrained)
        else:
            # Use timm for other backbones
            return self._create_timm_backbone(pretrained)
    
    def _create_resnet(self, pretrained: bool) -> Tuple[nn.Module, int]:
        """Create ResNet backbone."""
        resnet_configs = {
            'resnet18': (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, 512),
            'resnet34': (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, 512),
            'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, 2048),
            'resnet101': (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V2, 2048),
            'resnet152': (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V2, 2048)
        }
        
        if self.backbone_name not in resnet_configs:
            raise ValueError(f"Unknown ResNet variant: {self.backbone_name}")
        
        model_fn, weights, num_features = resnet_configs[self.backbone_name]
        
        if pretrained:
            model = model_fn(weights=weights)
        else:
            model = model_fn(weights=None)
        
        # Remove final avg pool and fc layer to get feature maps
        backbone = nn.Sequential(*list(model.children())[:-2])
        
        return backbone, num_features
    
    def _create_efficientnet(self, pretrained: bool) -> Tuple[nn.Module, int]:
        """Create EfficientNet backbone using timm."""
        # Use timm for EfficientNet
        model = timm.create_model(
            self.backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[-1]  # Only last feature map
        )
        
        # Get number of features from model config
        num_features = model.feature_info[-1]['num_chs']
        
        return model, num_features
    
    def _create_convnext(self, pretrained: bool) -> Tuple[nn.Module, int]:
        """Create ConvNeXt backbone."""
        convnext_configs = {
            'convnext_tiny': (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, 768),
            'convnext_small': (models.convnext_small, models.ConvNeXt_Small_Weights.IMAGENET1K_V1, 768),
            'convnext_base': (models.convnext_base, models.ConvNeXt_Base_Weights.IMAGENET1K_V1, 1024),
        }
        
        if self.backbone_name not in convnext_configs:
            # Try timm
            return self._create_timm_backbone(pretrained)
        
        model_fn, weights, num_features = convnext_configs[self.backbone_name]
        
        if pretrained:
            model = model_fn(weights=weights)
        else:
            model = model_fn(weights=None)
        
        # Remove classifier
        backbone = nn.Sequential(*list(model.children())[:-1])
        
        return backbone, num_features
    
    def _create_timm_backbone(self, pretrained: bool) -> Tuple[nn.Module, int]:
        """Create backbone using timm library for maximum flexibility."""
        model = timm.create_model(
            self.backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[-1]
        )
        
        num_features = model.feature_info[-1]['num_chs']
        
        return model, num_features
    
    def _freeze_layers(self, num_layers: int):
        """Freeze the first N layers of the backbone."""
        layers = list(self.backbone.children())
        for i, layer in enumerate(layers[:num_layers]):
            for param in layer.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Feature maps (B, num_features, H', W') if output_features=True
            Class logits (B, num_classes) if output_features=False
        """
        # Get feature maps from backbone
        if isinstance(self.backbone, nn.Sequential):
            features = self.backbone(x)
        else:
            # timm models return list
            features = self.backbone(x)
            if isinstance(features, list):
                features = features[-1]
        
        if self.output_features:
            if self.projection is not None:
                return self.projection(features)
            return features
        else:
            return self.classifier(features)
    
    def get_feature_info(self) -> Dict:
        """Get information about output features."""
        return {
            'num_features': self.num_features,
            'backbone': self.backbone_name,
            'output_features': self.output_features
        }


class MultiscaleCNNBackbone(nn.Module):
    """
    CNN backbone that extracts features at multiple scales.
    Useful for capturing both fine and coarse-grained patterns.
    """
    
    def __init__(
        self,
        backbone_name: str = 'resnet50',
        pretrained: bool = True,
        output_scales: List[int] = [2, 3, 4]  # Which layer outputs to use
    ):
        """
        Args:
            backbone_name: Name of the backbone
            pretrained: Use pretrained weights
            output_scales: List of layer indices to extract features from
        """
        super().__init__()
        
        self.output_scales = output_scales
        
        # Use timm for easy multi-scale feature extraction
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=output_scales
        )
        
        # Get channel info for each scale
        self.feature_info = self.backbone.feature_info
        self.num_features = [info['num_chs'] for info in self.feature_info]
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            List of feature maps at different scales
        """
        return self.backbone(x)


def get_cnn_backbone(
    name: str = 'resnet50',
    pretrained: bool = True,
    **kwargs
) -> CNNBackbone:
    """
    Factory function to create CNN backbone.
    
    Args:
        name: Backbone name
        pretrained: Use pretrained weights
        **kwargs: Additional arguments for CNNBackbone
        
    Returns:
        CNNBackbone instance
    """
    return CNNBackbone(
        backbone_name=name,
        pretrained=pretrained,
        **kwargs
    )


# Feature Pyramid Network for multi-scale features
class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for combining multi-scale features.
    Useful for detecting tumors of different sizes.
    """
    
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256
    ):
        """
        Args:
            in_channels_list: Number of channels for each input scale
            out_channels: Number of output channels for all scales
        """
        super().__init__()
        
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])
        
        self.out_channels = out_channels
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Combine multi-scale features.
        
        Args:
            features: List of feature maps from backbone
            
        Returns:
            List of combined feature maps (same scales)
        """
        # Lateral connections
        laterals = [
            conv(feat) for conv, feat in zip(self.lateral_convs, features)
        ]
        
        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            upsampled = nn.functional.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[-2:],
                mode='nearest'
            )
            laterals[i] = laterals[i] + upsampled
        
        # Final convolutions
        outputs = [
            conv(lat) for conv, lat in zip(self.fpn_convs, laterals)
        ]
        
        return outputs
