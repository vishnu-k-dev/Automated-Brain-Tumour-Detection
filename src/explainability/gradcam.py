"""
Grad-CAM implementation for CNN explainability.
Highlights regions most important for classification decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import cv2


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    Visualizes which regions of an image contribute to predictions.
    """
    
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Args:
            model: The neural network model
            target_layer: Layer to compute Grad-CAM for (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Auto-detect target layer if not provided
        if target_layer is None:
            self.target_layer = self._find_target_layer()
        
        self._register_hooks()
    
    def _find_target_layer(self) -> nn.Module:
        """Find the last convolutional layer."""
        target = None
        
        # Check if model has CNN backbone
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'cnn'):
            # Navigate to CNN backbone
            cnn = self.model.model.cnn.backbone
        elif hasattr(self.model, 'cnn'):
            cnn = self.model.cnn.backbone
        else:
            cnn = self.model
        
        # Find last conv layer
        for name, module in cnn.named_modules():
            if isinstance(module, nn.Conv2d):
                target = module
        
        if target is None:
            raise ValueError("Could not find convolutional layer for Grad-CAM")
        
        return target
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (uses predicted class if None)
            
        Returns:
            Grad-CAM heatmap as numpy array (H, W)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        if isinstance(output, dict):
            output = output['logits']
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Compute Grad-CAM
        gradients = self.gradients
        activations = self.activations
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # Only positive contributions
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input size
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        return cam.squeeze().cpu().numpy()


def apply_gradcam(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.
    
    Args:
        image: Original image (H, W, C) in range [0, 1] or [0, 255]
        heatmap: Grad-CAM heatmap (H, W) in range [0, 1]
        alpha: Blending factor
        colormap: OpenCV colormap
        
    Returns:
        Blended image
    """
    # Ensure image is in [0, 255]
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    
    # Resize heatmap to match image
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8),
        colormap
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    blended = (alpha * heatmap_colored + (1 - alpha) * image).astype(np.uint8)
    
    return blended


def visualize_gradcam(
    model: nn.Module,
    image: torch.Tensor,
    original_image: np.ndarray,
    class_names: List[str],
    target_class: Optional[int] = None,
    save_path: Optional[str] = None
):
    """
    Complete Grad-CAM visualization with prediction info.
    """
    gradcam = GradCAM(model)
    
    # Get prediction
    with torch.no_grad():
        output = model(image)
        if isinstance(output, dict):
            output = output['logits']
        probs = F.softmax(output, dim=-1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    
    # Generate heatmap
    if target_class is None:
        target_class = pred_class
    
    heatmap = gradcam.generate(image, target_class)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    if original_image.max() <= 1:
        original_image = (original_image * 255).astype(np.uint8)
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Heatmap
    im = axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    overlay = apply_gradcam(original_image, heatmap)
    axes[2].imshow(overlay)
    axes[2].set_title(f'Prediction: {class_names[pred_class]} ({confidence:.2%})')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig, heatmap
