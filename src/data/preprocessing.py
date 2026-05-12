"""
Preprocessing utilities for brain MRI images.
Includes normalization, skull stripping, and contrast enhancement.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union
import torch


def normalize_image(image: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize image intensities.
    
    Args:
        image: Input image array (H, W) or (H, W, C)
        method: Normalization method - 'zscore', 'minmax', or 'histogram'
        
    Returns:
        Normalized image array
    """
    if method == 'zscore':
        # Z-score normalization (standard)
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            normalized = (image - mean) / std
        else:
            normalized = image - mean
        # Clip to reasonable range and scale to [0, 1]
        normalized = np.clip(normalized, -3, 3)
        normalized = (normalized + 3) / 6
        
    elif method == 'minmax':
        # Min-max normalization to [0, 1]
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > min_val:
            normalized = (image - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(image)
            
    elif method == 'histogram':
        # Histogram equalization
        if len(image.shape) == 3:
            # Convert to LAB and equalize L channel
            lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
            normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        else:
            normalized = cv2.equalizeHist(image.astype(np.uint8)).astype(np.float32) / 255.0
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized.astype(np.float32)


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0, 
                     tile_size: int = 8) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Improves local contrast while limiting noise amplification.
    
    Args:
        image: Input image array (H, W) or (H, W, C)
        clip_limit: Clipping limit for CLAHE
        tile_size: Size of tiles for local histogram equalization
        
    Returns:
        Contrast-enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    
    if len(image.shape) == 3:
        # Apply CLAHE to each channel or convert to LAB
        if image.shape[2] == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        else:
            enhanced = np.stack([clahe.apply((c * 255).astype(np.uint8)).astype(np.float32) / 255.0 
                                for c in np.moveaxis(image, -1, 0)], axis=-1)
    else:
        enhanced = clahe.apply((image * 255).astype(np.uint8)).astype(np.float32) / 255.0
    
    return enhanced


def skull_strip(image: np.ndarray, threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple skull stripping using thresholding and morphological operations.
    For production, consider using HD-BET or other deep learning methods.
    
    Args:
        image: Input brain MRI image (H, W)
        threshold: Intensity threshold for brain mask
        
    Returns:
        Tuple of (skull-stripped image, brain mask)
    """
    # Normalize image for thresholding
    img_norm = normalize_image(image, 'minmax')
    
    # Binary threshold
    binary = (img_norm > threshold).astype(np.uint8)
    
    # Morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Fill holes
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Remove small components
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find largest connected component (the brain)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels > 1:
        # Find largest component (excluding background)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype(np.uint8)
    
    # Apply mask to original image
    skull_stripped = image * mask
    
    return skull_stripped, mask


def resize_image(image: np.ndarray, size: Tuple[int, int], 
                 keep_aspect: bool = False) -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        image: Input image array
        size: Target size (height, width)
        keep_aspect: Whether to keep aspect ratio (will pad if True)
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    target_h, target_w = size
    
    if keep_aspect:
        # Calculate scaling factor
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target size
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2
        
        if len(image.shape) == 3:
            result = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
            result[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        else:
            result = np.zeros((target_h, target_w), dtype=image.dtype)
            result[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    else:
        result = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    return result


def preprocess_image(
    image: Union[np.ndarray, Image.Image, str],
    size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
    normalize_method: str = 'zscore',
    enhance: bool = True,
    skull_strip_img: bool = False
) -> np.ndarray:
    """
    Complete preprocessing pipeline for brain MRI images.
    
    Args:
        image: Input image (array, PIL Image, or path)
        size: Target size (height, width)
        normalize: Whether to normalize intensities
        normalize_method: Normalization method
        enhance: Whether to apply contrast enhancement
        skull_strip_img: Whether to apply skull stripping
        
    Returns:
        Preprocessed image as numpy array (H, W, C)
    """
    # Load image if path
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    
    # Convert PIL to numpy
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure float32
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    # Apply skull stripping
    if skull_strip_img and len(image.shape) == 2:
        image, _ = skull_strip(image)
    
    # Apply contrast enhancement
    if enhance:
        image = enhance_contrast(image)
    
    # Resize
    image = resize_image(image, size)
    
    # Normalize
    if normalize:
        image = normalize_image(image, normalize_method)
    
    # Ensure 3 channels for consistency
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    
    return image


def extract_brain_roi(image: np.ndarray, padding: int = 10) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Extract the brain region of interest by finding bounding box.
    
    Args:
        image: Input brain MRI image
        padding: Padding around the ROI
        
    Returns:
        Tuple of (cropped image, bounding box coordinates)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (image * 255).astype(np.uint8)
    
    # Threshold
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        # Add padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        roi = image[y:y+h, x:x+w]
        return roi, (x, y, w, h)
    
    return image, (0, 0, image.shape[1], image.shape[0])


class ImagePreprocessor:
    """
    Reusable image preprocessor with configurable pipeline.
    """
    
    def __init__(
        self,
        size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        normalize_method: str = 'zscore',
        enhance: bool = True,
        skull_strip: bool = False
    ):
        self.size = size
        self.normalize = normalize
        self.normalize_method = normalize_method
        self.enhance = enhance
        self.skull_strip = skull_strip
    
    def __call__(self, image: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        return preprocess_image(
            image,
            size=self.size,
            normalize=self.normalize,
            normalize_method=self.normalize_method,
            enhance=self.enhance,
            skull_strip_img=self.skull_strip
        )
    
    def to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert preprocessed image to PyTorch tensor."""
        # HWC -> CHW
        if len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image).float()
