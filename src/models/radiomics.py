"""
Radiomics feature extraction for brain tumor analysis.
Combines hand-crafted texture/shape features with deep learning.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import cv2


class RadiomicsExtractor:
    """
    Extract hand-crafted radiomics features from brain MRI images.
    Features include texture, shape, and first-order statistics.
    """
    
    def __init__(self, feature_types: List[str] = None):
        self.feature_types = feature_types or ['firstorder', 'glcm', 'shape']
    
    def extract(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Extract all radiomics features from image."""
        if mask is None:
            mask = self._create_tumor_mask(image)
        
        features = []
        if 'firstorder' in self.feature_types:
            features.extend(self._extract_firstorder(image, mask))
        if 'glcm' in self.feature_types:
            features.extend(self._extract_glcm(image, mask))
        if 'shape' in self.feature_types:
            features.extend(self._extract_shape(mask))
        
        return np.array(features, dtype=np.float32)
    
    def _create_tumor_mask(self, image: np.ndarray) -> np.ndarray:
        """Create approximate tumor mask using thresholding."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return mask
    
    def _extract_firstorder(self, image: np.ndarray, mask: np.ndarray) -> List[float]:
        """Extract first-order statistics."""
        if len(image.shape) == 3:
            image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
        
        masked = image[mask > 0]
        if len(masked) == 0:
            return [0.0] * 10
        
        return [
            float(np.mean(masked)),
            float(np.std(masked)),
            float(np.min(masked)),
            float(np.max(masked)),
            float(np.median(masked)),
            float(np.percentile(masked, 25)),
            float(np.percentile(masked, 75)),
            float(np.sum(masked)),
            float(len(masked)),
            float(np.var(masked))
        ]
    
    def _extract_glcm(self, image: np.ndarray, mask: np.ndarray) -> List[float]:
        """Extract GLCM texture features."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        levels = 8
        gray = (gray // 32).astype(np.uint8)
        
        glcm = np.zeros((levels, levels), dtype=np.float32)
        h, w = gray.shape
        for i in range(h - 1):
            for j in range(w - 1):
                if mask[i, j] > 0 and mask[i, j+1] > 0:
                    glcm[gray[i, j], gray[i, j+1]] += 1
        
        glcm = glcm / (glcm.sum() + 1e-8)
        
        i_idx, j_idx = np.ogrid[:levels, :levels]
        contrast = np.sum(glcm * (i_idx - j_idx) ** 2)
        homogeneity = np.sum(glcm / (1 + np.abs(i_idx - j_idx)))
        energy = np.sum(glcm ** 2)
        correlation = np.sum(glcm * (i_idx - levels/2) * (j_idx - levels/2))
        entropy = -np.sum(glcm * np.log2(glcm + 1e-8))
        
        return [float(contrast), float(homogeneity), float(energy), 
                float(correlation), float(entropy)]
    
    def _extract_shape(self, mask: np.ndarray) -> List[float]:
        """Extract shape features."""
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return [0.0] * 7
        
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-8)
        
        x, y, w, h = cv2.boundingRect(largest)
        extent = area / (w * h + 1e-8)
        aspect_ratio = float(w) / (h + 1e-8)
        
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-8)
        
        return [float(area), float(perimeter), float(circularity),
                float(extent), float(aspect_ratio), float(solidity), float(len(largest))]


class LearnableRadiomics(nn.Module):
    """Learnable radiomics feature extraction using CNNs."""
    
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
            nn.Linear(64, feature_dim // 2)
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
            nn.Linear(64, feature_dim // 2)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
        
        self.feature_dim = feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        texture = self.texture_branch(x)
        shape = self.shape_branch(x)
        combined = torch.cat([texture, shape], dim=-1)
        return self.fusion(combined)


class RadiomicsIntegration(nn.Module):
    """Integrate hand-crafted and learned radiomics."""
    
    def __init__(self, handcrafted_dim: int = 22, learned_dim: int = 128,
                 output_dim: int = 128, in_channels: int = 3):
        super().__init__()
        
        self.learnable = LearnableRadiomics(in_channels, learned_dim)
        self.handcrafted_proj = nn.Linear(handcrafted_dim, learned_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(learned_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.output_dim = output_dim
    
    def forward(self, image: torch.Tensor, 
                handcrafted: Optional[torch.Tensor] = None) -> torch.Tensor:
        learned = self.learnable(image)
        
        if handcrafted is not None:
            hc_proj = self.handcrafted_proj(handcrafted)
            combined = torch.cat([learned, hc_proj], dim=-1)
            return self.fusion(combined)
        
        return learned
