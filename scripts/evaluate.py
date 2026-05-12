"""
Evaluation script for trained brain tumor classifier.
"""

import argparse
import yaml
import torch
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_data_loaders
from src.models import create_model
from src.evaluation import evaluate_model, get_classification_report
from src.evaluation.visualization import (
    plot_confusion_matrix, plot_roc_curves, plot_calibration_curve
)
from src.evaluation.metrics import compute_roc_curves, compute_calibration


def main():
    parser = argparse.ArgumentParser(description='Evaluate Brain Tumor Classifier')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override data directory')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which split to evaluate on')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.data_dir:
        config['data']['raw_dir'] = args.data_dir
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    loaders = get_data_loaders(
        data_dir=config['data']['raw_dir'],
        batch_size=config['training']['batch_size'],
        img_size=config['data']['img_size'],
        num_workers=config['hardware']['num_workers']
    )
    
    dataloader = loaders[args.split]
    print(f"Evaluating on {args.split} set: {len(dataloader.dataset)} samples")
    
    # Load model
    print("\nLoading model...")
    model = create_model(config)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Loaded checkpoint from epoch {epoch}")
    
    # Evaluate
    print("\nRunning evaluation...")
    class_names = config['data']['class_names']
    
    metrics = evaluate_model(
        model, dataloader, device,
        class_names=class_names,
        return_predictions=True
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
    print(f"F1 Score (weighted): {metrics['f1_weighted']:.4f}")
    print(f"ROC-AUC (OvR): {metrics['roc_auc_ovr']:.4f}")
    print(f"ROC-AUC (OvO): {metrics['roc_auc_ovo']:.4f}")
    
    print("\nPer-class metrics:")
    for name in class_names:
        print(f"  {name}:")
        print(f"    Precision: {metrics[f'precision_{name}']:.4f}")
        print(f"    Recall: {metrics[f'recall_{name}']:.4f}")
        print(f"    F1: {metrics[f'f1_{name}']:.4f}")
    
    print("\nClassification Report:")
    print(get_classification_report(
        metrics['predictions'],
        metrics['labels'],
        class_names
    ))
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics to JSON
    metrics_save = {k: v for k, v in metrics.items() 
                    if not isinstance(v, np.ndarray)}
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_save, f, indent=2, default=str)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names,
        save_path=str(output_dir / 'confusion_matrix.png'),
        normalize=True
    )
    
    # ROC curves
    roc_data = compute_roc_curves(
        metrics['probabilities'],
        metrics['labels'],
        config['data']['num_classes']
    )
    plot_roc_curves(
        roc_data,
        class_names,
        save_path=str(output_dir / 'roc_curves.png')
    )
    
    # Calibration curve
    calibration = compute_calibration(
        metrics['probabilities'],
        metrics['labels']
    )
    plot_calibration_curve(
        calibration,
        save_path=str(output_dir / 'calibration_curve.png')
    )
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    import numpy as np
    main()
