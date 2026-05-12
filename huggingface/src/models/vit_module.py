"""
Vision Transformer (ViT) module for global context learning.
Captures long-range dependencies in brain MRI images that CNNs miss.

This module is designed to work with CNN feature maps, not raw images,
enabling a hybrid CNN-ViT architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from einops import rearrange, repeat


class PatchEmbedding(nn.Module):
    """
    Convert CNN feature maps to patch embeddings for Vision Transformer.
    
    Unlike standard ViT which patches raw images, this module patches
    CNN feature maps, combining the benefits of both architectures.
    """
    
    def __init__(
        self,
        feature_size: int = 7,  # Size of CNN feature map (e.g., 7x7 for ResNet50)
        feature_dim: int = 2048,  # Channels of CNN feature map
        embed_dim: int = 512,  # Transformer embedding dimension
        patch_size: int = 1,  # How to group feature map locations
    ):
        """
        Args:
            feature_size: Spatial size of input feature maps
            feature_dim: Number of channels in feature maps
            embed_dim: Output embedding dimension
            patch_size: Size of patches to create (1 = each position is a token)
        """
        super().__init__()
        
        self.feature_size = feature_size
        self.patch_size = patch_size
        self.num_patches = (feature_size // patch_size) ** 2
        
        # Linear projection from feature channels to embedding dim
        if patch_size == 1:
            self.projection = nn.Linear(feature_dim, embed_dim)
        else:
            self.projection = nn.Conv2d(
                feature_dim, 
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size
            )
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )
        
        self.embed_dim = embed_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert feature maps to patch embeddings.
        
        Args:
            x: CNN feature maps (B, C, H, W)
            
        Returns:
            Patch embeddings with CLS token (B, num_patches + 1, embed_dim)
        """
        B = x.shape[0]
        
        if self.patch_size == 1:
            # Flatten spatial dimensions and project
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.projection(x)
        else:
            x = self.projection(x)
            x = rearrange(x, 'b c h w -> b (h w) c')
        
        # Prepend CLS token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embedding[:, :x.size(1)]
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    Core component of the Vision Transformer.
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1
    ):
        """
        Args:
            embed_dim: Input embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate for output
            attention_dropout: Dropout rate for attention weights
        """
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Combined QKV projection (more efficient)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Store attention weights for visualization
        self.attention_weights = None
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply multi-head self-attention.
        
        Args:
            x: Input tensor (B, N, D)
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor (B, N, D) and optionally attention weights
        """
        B, N, D = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Store for visualization
        self.attention_weights = attn.detach()
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        if return_attention:
            return x, attn
        return x, None


class TransformerBlock(nn.Module):
    """
    Single Transformer encoder block.
    Consists of Multi-Head Self-Attention + Feed-Forward Network.
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1
    ):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embed_dim
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
        """
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
            nn.Dropout(dropout)
        )
        
        # Store attention for visualization
        self.attention_weights = None
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with residual connections.
        
        Args:
            x: Input tensor (B, N, D)
            return_attention: Return attention weights
            
        Returns:
            Output tensor and optionally attention weights
        """
        # Self-attention with residual
        attn_out, attn = self.attn(self.norm1(x), return_attention)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        self.attention_weights = attn
        
        if return_attention:
            return x, attn
        return x, None


class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder.
    
    Processes patch embeddings through multiple transformer blocks
    to learn global context and long-range dependencies.
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1
    ):
        """
        Args:
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.depth = depth
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout
            )
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Store all attention weights for visualization
        self.attention_weights_all = []
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Process patch embeddings through transformer.
        
        Args:
            x: Patch embeddings (B, N, D)
            return_attention: Return all attention weights
            
        Returns:
            Encoded features (B, N, D) and optionally attention weights
        """
        self.attention_weights_all = []
        
        for block in self.blocks:
            x, attn = block(x, return_attention)
            if return_attention and attn is not None:
                self.attention_weights_all.append(attn)
        
        x = self.norm(x)
        
        if return_attention:
            return x, self.attention_weights_all
        return x, None
    
    def get_cls_token(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the CLS token representation after encoding.
        
        Args:
            x: Patch embeddings with CLS token (B, N+1, D)
            
        Returns:
            CLS token features (B, D)
        """
        x, _ = self.forward(x)
        return x[:, 0]  # First token is CLS


class CNNViTHybridEncoder(nn.Module):
    """
    Combined CNN-ViT encoder that processes images through both
    CNN backbone and Vision Transformer.
    
    This is the core of the hybrid architecture:
    1. CNN extracts local texture/shape features
    2. ViT captures global context and long-range dependencies
    """
    
    def __init__(
        self,
        cnn_feature_dim: int = 2048,
        cnn_feature_size: int = 7,
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        """
        Args:
            cnn_feature_dim: Channels from CNN backbone
            cnn_feature_size: Spatial size of CNN features
            embed_dim: ViT embedding dimension
            depth: Number of ViT blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP ratio in ViT
            dropout: Dropout rate
        """
        super().__init__()
        
        # Patch embedding from CNN features
        self.patch_embed = PatchEmbedding(
            feature_size=cnn_feature_size,
            feature_dim=cnn_feature_dim,
            embed_dim=embed_dim,
            patch_size=1
        )
        
        # Dropout after embedding
        self.embed_dropout = nn.Dropout(dropout)
        
        # ViT encoder
        self.vit_encoder = ViTEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        self.embed_dim = embed_dim
    
    def forward(
        self,
        cnn_features: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Process CNN features through ViT.
        
        Args:
            cnn_features: Feature maps from CNN (B, C, H, W)
            return_attention: Return attention weights
            
        Returns:
            CLS token features (B, embed_dim) and optionally attention weights
        """
        # Create patch embeddings
        x = self.patch_embed(cnn_features)
        x = self.embed_dropout(x)
        
        # Process through ViT
        x, attention = self.vit_encoder(x, return_attention)
        
        # Return CLS token
        cls_token = x[:, 0]
        
        if return_attention:
            return cls_token, attention
        return cls_token, None


class AttentionRollout:
    """
    Compute attention rollout for visualization.
    
    Recursively multiplies attention matrices to see
    where the model focuses across all layers.
    """
    
    def __init__(self, head_fusion: str = 'mean'):
        """
        Args:
            head_fusion: How to combine heads ('mean', 'max', 'min')
        """
        self.head_fusion = head_fusion
    
    def __call__(self, attention_weights: list) -> torch.Tensor:
        """
        Compute attention rollout.
        
        Args:
            attention_weights: List of attention matrices from each layer
                              Each has shape (B, heads, N, N)
        
        Returns:
            Rolled out attention (B, N, N)
        """
        result = torch.eye(attention_weights[0].shape[-1]).unsqueeze(0)
        result = result.to(attention_weights[0].device)
        
        for attention in attention_weights:
            # Fuse heads
            if self.head_fusion == 'mean':
                attention_heads_fused = attention.mean(dim=1)
            elif self.head_fusion == 'max':
                attention_heads_fused = attention.max(dim=1)[0]
            else:
                attention_heads_fused = attention.min(dim=1)[0]
            
            # Add identity for residual connection
            I = torch.eye(attention_heads_fused.shape[-1]).to(attention_heads_fused.device)
            attention_heads_fused = 0.5 * attention_heads_fused + 0.5 * I
            
            # Normalize rows
            attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)
            
            # Multiply with running result
            result = torch.matmul(attention_heads_fused, result)
        
        return result
