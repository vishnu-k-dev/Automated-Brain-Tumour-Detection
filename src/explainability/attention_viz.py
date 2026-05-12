"""
Attention visualization for Vision Transformer.
Shows global dependencies learned by the model.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple


def attention_rollout(
    attention_weights: List[torch.Tensor],
    discard_ratio: float = 0.0,
    head_fusion: str = 'mean'
) -> np.ndarray:
    """
    Compute attention rollout from ViT attention weights.
    
    Args:
        attention_weights: List of attention matrices from each layer
        discard_ratio: Ratio of lowest attention weights to discard
        head_fusion: How to fuse heads ('mean', 'max', 'min')
        
    Returns:
        Rolled out attention map
    """
    result = torch.eye(attention_weights[0].shape[-1])
    result = result.to(attention_weights[0].device)
    
    with torch.no_grad():
        for attention in attention_weights:
            # Fuse attention heads
            if head_fusion == 'mean':
                attention_fused = attention.mean(dim=1)
            elif head_fusion == 'max':
                attention_fused = attention.max(dim=1)[0]
            else:
                attention_fused = attention.min(dim=1)[0]
            
            # Discard low attention values
            if discard_ratio > 0:
                flat = attention_fused.view(-1)
                k = int(flat.size(0) * discard_ratio)
                if k > 0:
                    threshold = flat.kthvalue(k).values
                    attention_fused = attention_fused * (attention_fused > threshold).float()
            
            # Add residual connection
            I = torch.eye(attention_fused.shape[-1], device=attention_fused.device)
            attention_fused = 0.5 * attention_fused + 0.5 * I
            
            # Re-normalize
            attention_fused = attention_fused / attention_fused.sum(dim=-1, keepdim=True)
            
            # Multiply
            result = torch.matmul(attention_fused, result)
    
    # Get attention to CLS token
    mask = result[0, 0, 1:]  # Exclude CLS token itself
    
    return mask.cpu().numpy()


def visualize_attention(
    model: nn.Module,
    image: torch.Tensor,
    original_image: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
):
    """
    Visualize ViT attention maps.
    """
    model.eval()
    
    # Forward pass with attention
    with torch.no_grad():
        if hasattr(model, 'model'):
            output = model.model(image, return_attention=True)
            attention = output.get('attention', None)
            logits = output['logits']
        else:
            output = model(image, return_attention=True)
            attention = output.get('attention', None)
            logits = output['logits']
        
        probs = torch.softmax(logits, dim=-1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    
    if attention is None or len(attention) == 0:
        print("No attention weights available")
        return None
    
    # Compute attention rollout
    attn_map = attention_rollout(attention)
    
    # Reshape to grid (assuming 7x7 = 49 patches for 224x224 input)
    num_patches = int(np.sqrt(len(attn_map)))
    attn_map = attn_map.reshape(num_patches, num_patches)
    
    # Resize to image size
    import cv2
    attn_map_resized = cv2.resize(attn_map, (original_image.shape[1], original_image.shape[0]))
    
    # Normalize
    attn_map_resized = (attn_map_resized - attn_map_resized.min()) / (attn_map_resized.max() - attn_map_resized.min() + 1e-8)
    
    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    if original_image.max() <= 1:
        display_img = (original_image * 255).astype(np.uint8)
    else:
        display_img = original_image.astype(np.uint8)
    
    axes[0].imshow(display_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Raw attention map
    axes[1].imshow(attn_map, cmap='viridis')
    axes[1].set_title('Attention Map (Grid)')
    axes[1].axis('off')
    
    # Resized attention
    im = axes[2].imshow(attn_map_resized, cmap='hot')
    axes[2].set_title('Attention Map (Resized)')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Overlay
    overlay = display_img.copy().astype(np.float32)
    colored_map = plt.cm.jet(attn_map_resized)[:, :, :3] * 255
    overlay = 0.5 * overlay + 0.5 * colored_map
    overlay = overlay.astype(np.uint8)
    
    axes[3].imshow(overlay)
    axes[3].set_title(f'Pred: {class_names[pred_class]} ({confidence:.2%})')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig, attn_map_resized


def plot_attention_heads(
    attention: torch.Tensor,
    layer_idx: int = -1,
    save_path: Optional[str] = None
):
    """
    Visualize individual attention heads.
    """
    if layer_idx == -1:
        layer_idx = len(attention) - 1
    
    attn = attention[layer_idx][0]  # (num_heads, N, N)
    num_heads = attn.shape[0]
    
    # Plot each head
    cols = 4
    rows = (num_heads + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()
    
    for i in range(num_heads):
        # Get attention from CLS token to all patches
        head_attn = attn[i, 0, 1:].cpu().numpy()
        grid_size = int(np.sqrt(len(head_attn)))
        head_attn = head_attn.reshape(grid_size, grid_size)
        
        axes[i].imshow(head_attn, cmap='viridis')
        axes[i].set_title(f'Head {i+1}')
        axes[i].axis('off')
    
    # Hide unused axes
    for i in range(num_heads, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Attention Heads (Layer {layer_idx + 1})', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig
