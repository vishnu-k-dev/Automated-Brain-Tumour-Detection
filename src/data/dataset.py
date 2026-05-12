"""
Dataset classes for brain tumor MRI data.
Supports Kaggle Brain MRI dataset and BraTS format.
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Union
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold

from .preprocessing import preprocess_image
from .augmentation import get_train_transforms, get_val_transforms


class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for Brain Tumor MRI classification.
    
    Supports:
    - Kaggle Brain MRI dataset (folder structure with class subfolders)
    - Custom datasets with similar structure
    - Multimodal MRI (T1, T2, FLAIR) when available
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
        img_size: int = 224,
        preprocess: bool = True,
        multimodal: bool = False
    ):
        """
        Args:
            data_dir: Root directory containing class subfolders or split folders
            split: 'train', 'val', or 'test'
            transform: Albumentations transform to apply
            class_names: List of class names (auto-detected if None)
            img_size: Target image size
            preprocess: Whether to apply preprocessing
            multimodal: Whether to load multimodal MRI (T1, T2, FLAIR)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.img_size = img_size
        self.preprocess = preprocess
        self.multimodal = multimodal
        
        # Detect dataset structure
        self.images: List[str] = []
        self.labels: List[int] = []
        self.class_names: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        
        self._load_dataset(class_names)
        
        print(f"Loaded {len(self.images)} images for {split} split")
        print(f"Classes: {self.class_names}")
        print(f"Class distribution: {Counter(self.labels)}")
    
    def _load_dataset(self, class_names: Optional[List[str]] = None):
        """Load dataset from directory structure."""
        # Check for split-based structure (Training/Testing folders)
        split_dir = self.data_dir / ('Training' if self.split == 'train' else 'Testing')
        
        if split_dir.exists():
            data_path = split_dir
        else:
            # Assume flat structure with class folders
            data_path = self.data_dir
        
        # Get class names from folders
        if class_names is None:
            self.class_names = sorted([
                d.name for d in data_path.iterdir() 
                if d.is_dir() and not d.name.startswith('.')
            ])
        else:
            self.class_names = class_names
        
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Load image paths and labels
        for class_name in self.class_names:
            class_dir = data_path / class_name
            if not class_dir.exists():
                continue
                
            class_idx = self.class_to_idx[class_name]
            
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                    self.images.append(str(img_file))
                    self.labels.append(class_idx)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with 'image', 'label', and optionally 'path'
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Apply preprocessing
        if self.preprocess:
            image = preprocess_image(
                image,
                size=(self.img_size, self.img_size),
                normalize=False,  # Normalization done in transform
                enhance=True,
                skull_strip_img=False
            )
            # Ensure uint8 for albumentations
            image = (image * 255).astype(np.uint8)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Default: just convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'path': img_path
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced data."""
        class_counts = Counter(self.labels)
        total = len(self.labels)
        weights = [total / (len(class_counts) * class_counts[i]) for i in range(len(self.class_names))]
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_sample_weights(self) -> List[float]:
        """Get per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        return [class_weights[label].item() for label in self.labels]


class MultimodalBrainDataset(Dataset):
    """
    Dataset for multimodal brain MRI (T1, T2, FLAIR).
    Used with BraTS or similar multimodal datasets.
    """
    
    def __init__(
        self,
        data_dir: str,
        modalities: List[str] = ['t1', 't2', 'flair'],
        split: str = 'train',
        transform: Optional[Callable] = None,
        img_size: int = 224
    ):
        """
        Args:
            data_dir: Root directory containing patient folders
            modalities: List of modality names to load
            split: 'train', 'val', or 'test'
            transform: Albumentations transform
            img_size: Target image size
        """
        self.data_dir = Path(data_dir)
        self.modalities = modalities
        self.split = split
        self.transform = transform
        self.img_size = img_size
        
        self.samples: List[Dict] = []
        self._load_multimodal_data()
    
    def _load_multimodal_data(self):
        """Load multimodal MRI data from directory structure."""
        # Assume structure: data_dir/patient_id/{modality}.nii.gz or similar
        for patient_dir in self.data_dir.iterdir():
            if not patient_dir.is_dir():
                continue
            
            modality_files = {}
            for modality in self.modalities:
                # Try different naming conventions
                patterns = [
                    f"{modality}.nii.gz",
                    f"{modality}.nii",
                    f"*{modality}*.nii.gz",
                    f"*{modality}*.nii"
                ]
                
                for pattern in patterns:
                    matches = list(patient_dir.glob(pattern))
                    if matches:
                        modality_files[modality] = str(matches[0])
                        break
            
            if len(modality_files) == len(self.modalities):
                # TODO: Add label loading logic based on dataset structure
                self.samples.append({
                    'patient_id': patient_dir.name,
                    'modalities': modality_files,
                    'label': 0  # Placeholder - implement based on dataset
                })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a multimodal sample."""
        sample = self.samples[idx]
        
        images = []
        for modality in self.modalities:
            # Load NIfTI file (requires nibabel)
            try:
                import nibabel as nib
                nii = nib.load(sample['modalities'][modality])
                img = nii.get_fdata()
                
                # Get middle slice (2D) - adjust as needed
                slice_idx = img.shape[2] // 2
                img_2d = img[:, :, slice_idx]
                
                # Normalize
                img_2d = (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min() + 1e-8)
                img_2d = (img_2d * 255).astype(np.uint8)
                
                images.append(img_2d)
            except ImportError:
                print("nibabel required for NIfTI files. Install with: pip install nibabel")
                images.append(np.zeros((self.img_size, self.img_size), dtype=np.uint8))
        
        # Stack modalities as channels
        image = np.stack(images, axis=-1)
        
        # Resize to target size
        import cv2
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return {
            'image': image,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'patient_id': sample['patient_id']
        }


def get_data_loaders(
    data_dir: str,
    batch_size: int = 16,
    img_size: int = 224,
    num_workers: int = 0,  # Default to 0 for Windows compatibility
    val_split: float = 0.15,
    test_split: float = 0.15,
    use_weighted_sampler: bool = True,
    seed: int = 42
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir: Root data directory
        batch_size: Batch size for training
        img_size: Target image size
        num_workers: Number of data loading workers (use 0 on Windows)
        val_split: Validation split ratio
        test_split: Test split ratio
        use_weighted_sampler: Use weighted sampling for imbalanced classes
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train', 'val', and 'test' DataLoaders
    """
    data_path = Path(data_dir)
    
    # Check if dataset has pre-split structure
    has_training_folder = (data_path / 'Training').exists()
    has_testing_folder = (data_path / 'Testing').exists()
    
    train_transform = get_train_transforms(img_size=img_size)
    val_transform = get_val_transforms(img_size=img_size)
    
    if has_training_folder and has_testing_folder:
        # Use existing split - create separate dataset instances
        train_full_dataset = BrainTumorDataset(
            data_dir, split='train', transform=None, img_size=img_size
        )
        test_dataset = BrainTumorDataset(
            data_dir, split='test', transform=val_transform, img_size=img_size
        )
        
        # Create validation split from training data
        train_indices, val_indices = train_test_split(
            range(len(train_full_dataset)),
            test_size=val_split / (1 - test_split),
            stratify=train_full_dataset.labels,
            random_state=seed
        )
        
        # Use TransformSubset (defined at module level) for pickling compatibility
        train_dataset = TransformSubset(train_full_dataset, train_indices, train_transform)
        val_dataset = TransformSubset(train_full_dataset, val_indices, val_transform)
        
    else:
        # Single folder - create all splits
        full_dataset = BrainTumorDataset(
            data_dir, split='train', transform=None, img_size=img_size
        )
        
        # Stratified split
        train_val_indices, test_indices = train_test_split(
            range(len(full_dataset)),
            test_size=test_split,
            stratify=full_dataset.labels,
            random_state=seed
        )
        
        train_labels = [full_dataset.labels[i] for i in train_val_indices]
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_split / (1 - test_split),
            stratify=train_labels,
            random_state=seed
        )
        
        # Create datasets with appropriate transforms
        train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
        val_dataset = TransformSubset(full_dataset, val_indices, val_transform)
        test_dataset = TransformSubset(full_dataset, test_indices, val_transform)
    
    # Create samplers
    train_sampler = None
    shuffle = True
    
    if use_weighted_sampler and hasattr(train_dataset, 'dataset'):
        # Get sample weights for training
        original_dataset = train_dataset.dataset
        if hasattr(train_dataset, 'indices'):
            weights = [original_dataset.get_sample_weights()[i] for i in train_dataset.indices]
        else:
            weights = original_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False
    
    # Use num_workers=0 on Windows to avoid multiprocessing pickle issues
    import platform
    if platform.system() == 'Windows' and num_workers > 0:
        print("Note: Using num_workers=0 on Windows for compatibility")
        num_workers = 0
    
    # Create data loaders
    loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    }
    
    return loaders


class TransformSubset(torch.utils.data.Dataset):
    """Subset with custom transform."""
    
    def __init__(
        self,
        dataset: BrainTumorDataset,
        indices: List[int],
        transform: Optional[Callable] = None
    ):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        original_idx = self.indices[idx]
        img_path = self.dataset.images[original_idx]
        label = self.dataset.labels[original_idx]
        
        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.dataset.preprocess:
            image = preprocess_image(
                image,
                size=(self.dataset.img_size, self.dataset.img_size),
                normalize=False,
                enhance=True
            )
            image = (image * 255).astype(np.uint8)
        
        # Apply transform
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'path': img_path
        }


def get_cross_validation_splits(
    data_dir: str,
    n_splits: int = 5,
    seed: int = 42
) -> List[Tuple[List[int], List[int]]]:
    """
    Generate stratified K-fold cross-validation splits.
    
    Args:
        data_dir: Root data directory
        n_splits: Number of folds
        seed: Random seed
        
    Returns:
        List of (train_indices, val_indices) tuples
    """
    dataset = BrainTumorDataset(data_dir, split='train', transform=None)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    splits = []
    for train_idx, val_idx in skf.split(dataset.images, dataset.labels):
        splits.append((list(train_idx), list(val_idx)))
    
    return splits
