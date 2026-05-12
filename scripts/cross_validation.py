"""
K-Fold Cross-Validation for Brain Tumor Classification.
Provides mean ± std metrics with confidence intervals for publication.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import yaml
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import BrainTumorDataset, TransformSubset
from src.data.augmentation import get_train_transforms, get_val_transforms
from src.models import create_model
from src.training.trainer import train_epoch, validate_epoch
from src.training.losses import get_loss_function
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler


def run_cross_validation(
    data_dir: str,
    config_path: str,
    n_folds: int = 5,
    epochs_per_fold: int = 30,
    seed: int = 42
) -> Dict:
    """
    Run K-fold cross-validation.
    
    Args:
        data_dir: Path to data directory
        config_path: Path to config file
        n_folds: Number of folds
        epochs_per_fold: Training epochs per fold
        seed: Random seed
        
    Returns:
        Dictionary with fold results and summary statistics
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Running {n_folds}-fold cross-validation with {epochs_per_fold} epochs per fold")
    
    # Load full dataset
    print("\nLoading dataset...")
    full_dataset = BrainTumorDataset(
        data_dir, split='train', transform=None,
        img_size=config['data']['img_size']
    )
    
    # Get transforms
    img_size = config['data']['img_size']
    train_transform = get_train_transforms(img_size)
    val_transform = get_val_transforms(img_size)
    
    # Setup cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # Store results for each fold
    fold_results = []
    
    print("\n" + "="*60)
    print(f"STARTING {n_folds}-FOLD CROSS-VALIDATION")
    print("="*60)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(full_dataset.images, full_dataset.labels)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'='*60}")
        print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
        
        # Create fold datasets
        train_dataset = TransformSubset(full_dataset, list(train_idx), train_transform)
        val_dataset = TransformSubset(full_dataset, list(val_idx), val_transform)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=config['training']['batch_size'],
            shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config['training']['batch_size'],
            shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available()
        )
        
        # Create fresh model for each fold
        model = create_model(config).to(device)
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_per_fold
        )
        criterion = get_loss_function(config)
        scaler = GradScaler() if config['training'].get('mixed_precision', True) else None
        
        # Train for this fold
        best_f1 = 0
        best_metrics = None
        
        for epoch in range(epochs_per_fold):
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
            
            # Track best
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                best_metrics = val_metrics.copy()
            
            if (epoch + 1) % 10 == 0 or epoch == epochs_per_fold - 1:
                print(f"  Epoch {epoch+1}/{epochs_per_fold} - "
                      f"Val Acc: {val_metrics['accuracy']:.4f}, "
                      f"Val F1: {val_metrics['f1']:.4f}")
        
        # Get final evaluation with probabilities for ROC-AUC
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
        
        # Calculate final metrics
        all_probs = np.array(all_probs)
        try:
            roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
        except:
            roc_auc = 0.0
        
        fold_result = {
            'fold': fold + 1,
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
            'f1_macro': f1_score(all_labels, all_preds, average='macro'),
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'roc_auc': roc_auc
        }
        
        fold_results.append(fold_result)
        
        print(f"\n  Fold {fold + 1} Results:")
        print(f"    Accuracy:  {fold_result['accuracy']:.4f}")
        print(f"    F1 Score:  {fold_result['f1_weighted']:.4f}")
        print(f"    ROC-AUC:   {fold_result['roc_auc']:.4f}")
    
    # Calculate summary statistics
    metrics = ['accuracy', 'f1_weighted', 'f1_macro', 'precision', 'recall', 'roc_auc']
    summary = {}
    
    for metric in metrics:
        values = [r[metric] for r in fold_results]
        summary[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            '95_ci': 1.96 * np.std(values) / np.sqrt(n_folds)  # 95% CI
        }
    
    # Print summary
    print("\n" + "="*60)
    print("CROSS-VALIDATION SUMMARY")
    print("="*60)
    print(f"\n{n_folds}-Fold Cross-Validation Results (Mean ± Std):\n")
    
    for metric in metrics:
        s = summary[metric]
        print(f"  {metric.replace('_', ' ').title():15} "
              f"{s['mean']*100:6.2f}% ± {s['std']*100:.2f}% "
              f"(95% CI: ±{s['95_ci']*100:.2f}%)")
    
    print("\n" + "-"*60)
    print("Publication-Ready Format:")
    print("-"*60)
    print(f"\nAccuracy: {summary['accuracy']['mean']*100:.2f}% ± {summary['accuracy']['std']*100:.2f}%")
    print(f"F1-Score: {summary['f1_weighted']['mean']*100:.2f}% ± {summary['f1_weighted']['std']*100:.2f}%")
    print(f"ROC-AUC:  {summary['roc_auc']['mean']*100:.2f}% ± {summary['roc_auc']['std']*100:.2f}%")
    
    # Save results
    results = {
        'n_folds': n_folds,
        'epochs_per_fold': epochs_per_fold,
        'fold_results': fold_results,
        'summary': summary,
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = Path('results/cross_validation_results.json')
    results_path.parent.mkdir(exist_ok=True)
    
    # Convert numpy types for JSON serialization
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
    
    with open(results_path, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='K-Fold Cross-Validation')
    parser.add_argument('--data-dir', type=str, default='src/data/raw',
                        help='Path to data directory')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--folds', type=int, default=5,
                        help='Number of folds')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Epochs per fold')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    run_cross_validation(
        data_dir=args.data_dir,
        config_path=args.config,
        n_folds=args.folds,
        epochs_per_fold=args.epochs,
        seed=args.seed
    )
