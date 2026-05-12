"""
Feature fusion modules for combining different feature sources.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class MultimodalFusion(nn.Module):
    """Fuse multiple MRI modalities into a unified representation."""
    
    def __init__(self, in_channels: int = 1, num_modalities: int = 3,
                 out_channels: int = 3, fusion_type: str = 'attention'):
        super().__init__()
        self.fusion_type = fusion_type
        self.num_modalities = num_modalities
        
        if fusion_type == 'concat':
            self.fusion = nn.Conv2d(in_channels * num_modalities, out_channels, 1)
        elif fusion_type == 'attention':
            hidden_dim = out_channels * 2
            self.query = nn.Conv2d(in_channels * num_modalities, hidden_dim, 1)
            self.key = nn.Conv2d(in_channels * num_modalities, hidden_dim, 1)
            self.value = nn.Conv2d(in_channels * num_modalities, out_channels, 1)
        elif fusion_type == 'gated':
            self.gates = nn.ModuleList([
                nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())
                for _ in range(num_modalities)
            ])
            self.projection = nn.Conv2d(in_channels * num_modalities, out_channels, 1)
    
    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        if self.fusion_type == 'concat':
            return self.fusion(torch.cat(modalities, dim=1))
        elif self.fusion_type == 'attention':
            x = torch.cat(modalities, dim=1)
            B, C, H, W = x.shape
            q = self.query(x).view(B, -1, H * W)
            k = self.key(x).view(B, -1, H * W)
            v = self.value(x).view(B, -1, H * W)
            attn = F.softmax(torch.bmm(q.transpose(1, 2), k) / (q.shape[1] ** 0.5), dim=-1)
            return torch.bmm(v, attn.transpose(1, 2)).view(B, -1, H, W)
        elif self.fusion_type == 'gated':
            gated = [gate(m) * m for gate, m in zip(self.gates, modalities)]
            return self.projection(torch.cat(gated, dim=1))


class FeatureFusion(nn.Module):
    """Fuse CNN, ViT, and radiomics features."""
    
    def __init__(self, cnn_dim: int = 2048, vit_dim: int = 512,
                 radiomics_dim: int = 128, output_dim: int = 512,
                 fusion_type: str = 'concat', use_radiomics: bool = True):
        super().__init__()
        self.use_radiomics = use_radiomics
        self.fusion_type = fusion_type
        
        total_dim = cnn_dim + vit_dim + (radiomics_dim if use_radiomics else 0)
        
        if fusion_type == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(total_dim, output_dim * 2),
                nn.LayerNorm(output_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim * 2, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            )
        elif fusion_type == 'attention':
            self.cnn_proj = nn.Linear(cnn_dim, output_dim)
            self.vit_proj = nn.Linear(vit_dim, output_dim)
            if use_radiomics:
                self.rad_proj = nn.Linear(radiomics_dim, output_dim)
            self.cross_attn = nn.MultiheadAttention(output_dim, 8, batch_first=True)
            self.norm = nn.LayerNorm(output_dim)
            self.final_fc = nn.Linear(output_dim, output_dim)
        
        self.output_dim = output_dim
    
    def forward(self, cnn_features: torch.Tensor, vit_features: torch.Tensor,
                radiomics_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.fusion_type == 'concat':
            if self.use_radiomics and radiomics_features is not None:
                x = torch.cat([cnn_features, vit_features, radiomics_features], dim=-1)
            else:
                x = torch.cat([cnn_features, vit_features], dim=-1)
            return self.fusion(x)
        elif self.fusion_type == 'attention':
            cnn_proj = self.cnn_proj(cnn_features).unsqueeze(1)
            vit_proj = self.vit_proj(vit_features).unsqueeze(1)
            if self.use_radiomics and radiomics_features is not None:
                rad_proj = self.rad_proj(radiomics_features).unsqueeze(1)
                tokens = torch.cat([cnn_proj, vit_proj, rad_proj], dim=1)
            else:
                tokens = torch.cat([cnn_proj, vit_proj], dim=1)
            attn_out, _ = self.cross_attn(tokens, tokens, tokens)
            return self.final_fc(self.norm(attn_out.mean(dim=1)))


class AdaptiveFusion(nn.Module):
    """Adaptive fusion with learned weights."""
    
    def __init__(self, feature_dims: List[int], output_dim: int = 512):
        super().__init__()
        self.num_sources = len(feature_dims)
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, output_dim), nn.LayerNorm(output_dim), nn.GELU())
            for dim in feature_dims
        ])
        self.gate = nn.Sequential(
            nn.Linear(sum(feature_dims), output_dim), nn.GELU(),
            nn.Linear(output_dim, self.num_sources), nn.Softmax(dim=-1)
        )
        self.output_dim = output_dim
    
    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = self.gate(torch.cat(features, dim=-1))
        projected = [proj(f) for proj, f in zip(self.projections, features)]
        stacked = torch.stack(projected, dim=1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return fused, weights
