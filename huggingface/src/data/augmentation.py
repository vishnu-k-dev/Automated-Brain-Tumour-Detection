"""
Data augmentation transforms for brain MRI images.
Uses albumentations library for efficient augmentations.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from typing import Dict, Optional, Tuple
import torch


def get_train_transforms(
    img_size: int = 224,
    rotation_range: int = 15,
    horizontal_flip: bool = True,
    vertical_flip: bool = False,
    elastic_deform: bool = True,
    brightness_range: float = 0.1,
    p_augment: float = 0.5
) -> A.Compose:
    """
    Get training data augmentation transforms.
    
    Args:
        img_size: Target image size
        rotation_range: Maximum rotation angle in degrees
        horizontal_flip: Enable horizontal flipping
        vertical_flip: Enable vertical flipping
        elastic_deform: Enable elastic deformation
        brightness_range: Range for brightness adjustment
        p_augment: Probability of applying each augmentation
        
    Returns:
        Albumentations Compose object
    """
    transforms_list = [
        # Resize to target size
        A.Resize(img_size, img_size),
        
        # Geometric transforms
        A.Rotate(limit=rotation_range, p=p_augment, border_mode=cv2.BORDER_CONSTANT),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=0,
            p=p_augment,
            border_mode=cv2.BORDER_CONSTANT
        ),
    ]
    
    if horizontal_flip:
        transforms_list.append(A.HorizontalFlip(p=0.5))
    
    if vertical_flip:
        transforms_list.append(A.VerticalFlip(p=0.5))
    
    if elastic_deform:
        transforms_list.append(
            A.ElasticTransform(
                alpha=50,
                sigma=5,
                p=p_augment * 0.5,
                border_mode=cv2.BORDER_CONSTANT
            )
        )
    
    # Intensity transforms
    transforms_list.extend([
        A.RandomBrightnessContrast(
            brightness_limit=brightness_range,
            contrast_limit=brightness_range,
            p=p_augment
        ),
        A.GaussNoise(std_range=(0.02, 0.1), p=p_augment * 0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=p_augment * 0.2),
    ])
    
    # Optional: Grid distortion for medical images
    transforms_list.append(
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=p_augment * 0.3)
    )
    
    # Normalize and convert to tensor
    transforms_list.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return A.Compose(transforms_list)


def get_val_transforms(img_size: int = 224) -> A.Compose:
    """
    Get validation/test data transforms (no augmentation).
    
    Args:
        img_size: Target image size
        
    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


class MixupCutmix:
    """
    Implements Mixup and Cutmix data augmentation.
    These improve model generalization by creating interpolated samples.
    """
    
    def __init__(
        self,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        mixup_prob: float = 0.5,
        cutmix_prob: float = 0.0,
        num_classes: int = 4
    ):
        """
        Args:
            mixup_alpha: Beta distribution parameter for mixup
            cutmix_alpha: Beta distribution parameter for cutmix
            mixup_prob: Probability of applying mixup
            cutmix_prob: Probability of applying cutmix
            num_classes: Number of classes for one-hot encoding
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        self.num_classes = num_classes
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply mixup or cutmix to a batch.
        
        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B,) - will be converted to one-hot
            
        Returns:
            Augmented images and mixed labels
        """
        batch_size = images.size(0)
        
        # Convert labels to one-hot
        labels_one_hot = torch.zeros(batch_size, self.num_classes, device=labels.device)
        labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
        
        # Decide which augmentation to apply
        rand = np.random.random()
        
        if rand < self.mixup_prob:
            return self._mixup(images, labels_one_hot)
        elif rand < self.mixup_prob + self.cutmix_prob:
            return self._cutmix(images, labels_one_hot)
        else:
            return images, labels_one_hot
    
    def _mixup(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mixup augmentation."""
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        # Random permutation indices
        indices = torch.randperm(images.size(0), device=images.device)
        
        # Mix images and labels
        mixed_images = lam * images + (1 - lam) * images[indices]
        mixed_labels = lam * labels + (1 - lam) * labels[indices]
        
        return mixed_images, mixed_labels
    
    def _cutmix(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply cutmix augmentation."""
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        
        batch_size, _, h, w = images.size()
        indices = torch.randperm(batch_size, device=images.device)
        
        # Get cutmix bounding box
        bbx1, bby1, bbx2, bby2 = self._get_cutmix_bbox(h, w, lam)
        
        # Apply cutmix
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[indices, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (h * w))
        
        mixed_labels = lam * labels + (1 - lam) * labels[indices]
        
        return mixed_images, mixed_labels
    
    def _get_cutmix_bbox(
        self,
        h: int,
        w: int,
        lam: float
    ) -> Tuple[int, int, int, int]:
        """Get random bounding box for cutmix."""
        cut_ratio = np.sqrt(1 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        
        # Random center position
        cy = np.random.randint(h)
        cx = np.random.randint(w)
        
        # Calculate box coordinates
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        return bbx1, bby1, bbx2, bby2


class TestTimeAugmentation:
    """
    Test Time Augmentation (TTA) for improved prediction robustness.
    Applies multiple augmentations at inference and averages predictions.
    """
    
    def __init__(self, img_size: int = 224, num_augmentations: int = 5):
        """
        Args:
            img_size: Target image size
            num_augmentations: Number of augmented versions to create
        """
        self.img_size = img_size
        self.num_augmentations = num_augmentations
        
        # TTA transforms (gentle augmentations)
        self.transforms = [
            A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            A.Compose([
                A.Resize(img_size, img_size),
                A.Rotate(limit=10, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            A.Compose([
                A.Resize(img_size, img_size),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            A.Compose([
                A.Resize(img_size + 32, img_size + 32),
                A.CenterCrop(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
        ]
    
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """
        Apply TTA to an image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Stacked tensor of augmented images (num_augmentations, C, H, W)
        """
        augmented_images = []
        
        for i, transform in enumerate(self.transforms[:self.num_augmentations]):
            augmented = transform(image=image)['image']
            augmented_images.append(augmented)
        
        return torch.stack(augmented_images)
