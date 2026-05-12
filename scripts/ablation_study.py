"""
Ablation Study for Brain Tumor Classification.
Compares different model configurations to show component contributions.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
import json
from datetime import datetime
import timm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import BrainTumorDataset, TransformSubset
from src.data.augmentation import get_train_transforms, get_val_transforms
from src.training.trainer import train_epoch, validate_epoch
from src.training.losses import get_loss_function
from src.models import create_model
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from sklearn.model_selection import train_test_split


class CNNOnlyModel(nn.Module):
    """ResNet50 CNN-only baseline."""
    
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('resnet50', pretrained=pretrained, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class ViTOnlyModel(nn.Module):
    """Vision Transformer only baseline."""
    
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
    
    def forward(self, x):
        return self.vit(x)


class EfficientNetModel(nn.Module):
    """EfficientNet-B0 baseline."""
    
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)


def create_ablation_model(model_type: str, config: Dict) -> nn.Module:
    """Create model for ablation study."""
    num_classes = config['data']['num_classes']
    
    if model_type == 'cnn_only':
        return CNNOnlyModel(num_classes=num_classes)
    elif model_type == 'vit_only':
        return ViTOnlyModel(num_classes=num_classes)
    elif model_type == 'efficientnet':
        return EfficientNetModel(num_classes=num_classes)
    elif model_type == 'hybrid':
        return create_model(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    device: torch.device,
    epochs: int = 20
) -> Dict:
    """Train model and return metrics."""
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = get_loss_function(config)
    scaler = GradScaler() if config['training'].get('mixed_precision', True) else None
    
    best_f1 = 0
    best_metrics = None
    
    for epoch in range(epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            device, scaler, None,
            config['training'].get('gradient_clip', 1.0)
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, device,
            config['data']['num_classes']
        )
        
        scheduler.step()
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_metrics = val_metrics.copy()
        
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs} - Val Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
    
    # Final evaluation with probabilities
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['label']
            
            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs['logits']
            
            probs = torch.softmax(outputs, dim=-1)
            preds = probs.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_probs = np.array(all_probs)
    
    try:
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    except:
        roc_auc = 0.0
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_score': f1_score(all_labels, all_preds, average='weighted'),
        'roc_auc': roc_auc,
        'total_params': total_params,
        'trainable_params': trainable_params
    }


def run_ablation_study(
    data_dir: str,
    config_path: str,
    epochs_per_model: int = 20,
    seed: int = 42
):
    """Run ablation study comparing different model configurations."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load dataset
    print("\nLoading dataset...")
    full_dataset = BrainTumorDataset(
        data_dir, split='train', transform=None,
        img_size=config['data']['img_size']
    )
    
    # Split data
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)),
        test_size=0.2,
        stratify=full_dataset.labels,
        random_state=seed
    )
    
    img_size = config['data']['img_size']
    train_dataset = TransformSubset(full_dataset, list(train_idx), get_train_transforms(img_size))
    val_dataset = TransformSubset(full_dataset, list(val_idx), get_val_transforms(img_size))
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['training']['batch_size'],
        shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['training']['batch_size'],
        shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available()
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Models to compare
    models = {
        'CNN Only (ResNet50)': 'cnn_only',
        'ViT Only (ViT-B/16)': 'vit_only',
        'EfficientNet-B0': 'efficientnet',
        'Hybrid CNN-ViT (Ours)': 'hybrid'
    }
    
    results = {}
    
    print("\n" + "="*70)
    print("ABLATION STUDY")
    print("="*70)
    
    for name, model_type in models.items():
        print(f"\n{'='*70}")
        print(f"Training: {name}")
        print(f"{'='*70}")
        
        try:
            model = create_ablation_model(model_type, config)
            metrics = train_and_evaluate(
                model, train_loader, val_loader, config, device, epochs_per_model
            )
            results[name] = metrics
            
            print(f"\n  Results for {name}:")
            print(f"    Accuracy:   {metrics['accuracy']*100:.2f}%")
            print(f"    F1-Score:   {metrics['f1_score']*100:.2f}%")
            print(f"    ROC-AUC:    {metrics['roc_auc']*100:.2f}%")
            print(f"    Parameters: {metrics['total_params']/1e6:.1f}M")
            
        except Exception as e:
            print(f"  Error training {name}: {e}")
            results[name] = {'error': str(e)}
    
    # Print comparison table
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)
    print(f"\n{'Model':<30} {'Accuracy':>10} {'F1-Score':>10} {'ROC-AUC':>10} {'Params':>10}")
    print("-"*70)
    
    for name, metrics in results.items():
        if 'error' not in metrics:
            print(f"{name:<30} {metrics['accuracy']*100:>9.2f}% {metrics['f1_score']*100:>9.2f}% "
                  f"{metrics['roc_auc']*100:>9.2f}% {metrics['total_params']/1e6:>9.1f}M")
    
    # Calculate improvements
    if 'Hybrid CNN-ViT (Ours)' in results and 'error' not in results['Hybrid CNN-ViT (Ours)']:
        hybrid_acc = results['Hybrid CNN-ViT (Ours)']['accuracy']
        
        print("\n" + "-"*70)
        print("Improvement over baselines:")
        print("-"*70)
        
        for name, metrics in results.items():
            if name != 'Hybrid CNN-ViT (Ours)' and 'error' not in metrics:
                improvement = (hybrid_acc - metrics['accuracy']) * 100
                print(f"  vs {name}: +{improvement:.2f}%")
    
    # Save results
    output = {
        'models': results,
        'epochs_per_model': epochs_per_model,
        'seed': seed,
        'timestamp': datetime.now().isoformat()
    }
    
    # Convert numpy types
    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj
    
    results_path = Path('results/ablation_study_results.json')
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(convert_numpy(output), f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ablation Study')
    parser.add_argument('--data-dir', type=str, default='src/data/raw',
                        help='Path to data directory')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Epochs per model')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    run_ablation_study(
        data_dir=args.data_dir,
        config_path=args.config,
        epochs_per_model=args.epochs,
        seed=args.seed
    )
