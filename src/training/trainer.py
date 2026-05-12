"""
Main training loop for brain tumor classification.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Optional, Callable
import time
from pathlib import Path
from tqdm import tqdm
import json

from .losses import get_loss_function
from ..data.augmentation import MixupCutmix


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    mixup: Optional[MixupCutmix] = None,
    gradient_clip: float = 1.0
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Apply mixup/cutmix
        if mixup is not None:
            images, labels = mixup(images, labels)
            use_soft_labels = True
        else:
            use_soft_labels = False
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if scaler is not None:
            with autocast():
                outputs = model(images)
                if isinstance(outputs, dict):
                    outputs = outputs['logits']
                if use_soft_labels:
                    loss = -(labels * torch.log_softmax(outputs, dim=-1)).sum(dim=-1).mean()
                else:
                    loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs['logits']
            if use_soft_labels:
                loss = -(labels * torch.log_softmax(outputs, dim=-1)).sum(dim=-1).mean()
            else:
                loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        
        total_loss += loss.item()
        
        if not use_soft_labels:
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    metrics = {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total if total > 0 else 0
    }
    
    return metrics


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 4
) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs['logits']
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    metrics = {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    
    return metrics


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
    
    def _is_improvement(self, score: float) -> bool:
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta


class Trainer:
    """Complete training pipeline."""
    
    def __init__(self, model: nn.Module, config: Dict, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        train_config = config.get('training', {})
        
        # Optimizer
        self.optimizer = self._create_optimizer(train_config)
        self.scheduler = self._create_scheduler(train_config)
        
        # Loss function
        self.criterion = get_loss_function(config)
        
        # Mixed precision
        self.use_amp = train_config.get('mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Mixup
        mixup_alpha = config.get('augmentation', {}).get('mixup_alpha', 0)
        if mixup_alpha > 0:
            self.mixup = MixupCutmix(mixup_alpha=mixup_alpha,
                                     num_classes=config['data']['num_classes'])
        else:
            self.mixup = None
        
        # Early stopping
        es_config = train_config.get('early_stopping', {})
        if es_config.get('enabled', True):
            self.early_stopping = EarlyStopping(
                patience=es_config.get('patience', 15),
                min_delta=es_config.get('min_delta', 0.001),
                mode='max'
            )
        else:
            self.early_stopping = None
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint', {}).get('save_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_metric = 0
        self.history = {'train': [], 'val': []}
    
    def _create_optimizer(self, config: Dict) -> torch.optim.Optimizer:
        opt_name = config.get('optimizer', 'adamw').lower()
        lr = config.get('learning_rate', 1e-4)
        wd = config.get('weight_decay', 0.01)
        
        if opt_name == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            return torch.optim.SGD(self.model.parameters(), lr=lr, 
                                   momentum=0.9, weight_decay=wd)
    
    def _create_scheduler(self, config: Dict):
        sched_name = config.get('scheduler', 'cosine').lower()
        epochs = config.get('epochs', 100)
        warmup = config.get('warmup_epochs', 5)
        
        # Ensure T_max is at least 1
        t_max = max(epochs - warmup, 1)
        
        if sched_name == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, t_max)
        elif sched_name == 'step':
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        else:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=5)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Full training loop."""
        epochs = self.config.get('training', {}).get('epochs', 100)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_metrics = train_epoch(
                self.model, train_loader, self.criterion, self.optimizer,
                self.device, self.scaler, self.mixup,
                self.config.get('training', {}).get('gradient_clip', 1.0)
            )
            
            # Validate
            val_metrics = validate_epoch(
                self.model, val_loader, self.criterion, self.device,
                self.config['data']['num_classes']
            )
            
            # Update scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['f1'])
            else:
                self.scheduler.step()
            
            # Log
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}")
            
            self.history['train'].append(train_metrics)
            self.history['val'].append(val_metrics)
            
            # Save best model
            if val_metrics['f1'] > self.best_metric:
                self.best_metric = val_metrics['f1']
                self.save_checkpoint('best_model.pth', epoch, val_metrics)
                print(f"Saved best model with F1: {self.best_metric:.4f}")
            
            # Early stopping
            if self.early_stopping and self.early_stopping(val_metrics['f1']):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Save final model
        self.save_checkpoint('final_model.pth', epoch, val_metrics)
        
        return self.history
    
    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})
