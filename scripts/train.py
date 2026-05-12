"""
Main training script for brain tumor classification.
"""

import argparse
import yaml
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_data_loaders
from src.models import BrainTumorClassifier, create_model
from src.training import Trainer
from src.training.ssl_trainer import pretrain_ssl


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train Brain Tumor Classifier')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override data directory from config')
    parser.add_argument('--pretrain', action='store_true',
                        help='Run self-supervised pre-training first')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_dir:
        config['data']['raw_dir'] = args.data_dir
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Set random seed
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Load data
    print("\nLoading data...")
    data_dir = config['data']['raw_dir']
    loaders = get_data_loaders(
        data_dir=data_dir,
        batch_size=config['training']['batch_size'],
        img_size=config['data']['img_size'],
        num_workers=config['hardware']['num_workers'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split']
    )
    
    print(f"Train samples: {len(loaders['train'].dataset)}")
    print(f"Val samples: {len(loaders['val'].dataset)}")
    print(f"Test samples: {len(loaders['test'].dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Self-supervised pre-training
    if args.pretrain and config.get('ssl', {}).get('enabled', False):
        print("\nStarting self-supervised pre-training...")
        model = pretrain_ssl(model, loaders['train'], config, device)
    
    # Resume from checkpoint
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Training
    print("\nStarting training...")
    trainer = Trainer(model, config, device)
    history = trainer.train(loaders['train'], loaders['val'])
    
    # Final evaluation on test set
    print("\n" + "=" * 50)
    print("Final Evaluation on Test Set")
    print("=" * 50)
    
    from src.evaluation import evaluate_model, get_classification_report
    from src.evaluation.visualization import plot_confusion_matrix, plot_roc_curves
    from src.evaluation.metrics import compute_roc_curves
    
    # Load best model
    best_checkpoint = trainer.checkpoint_dir / 'best_model.pth'
    model.load_state_dict(torch.load(best_checkpoint)['model_state_dict'])
    
    # Evaluate
    metrics = evaluate_model(
        model, loaders['test'], device,
        class_names=config['data']['class_names']
    )
    
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"Test ROC-AUC: {metrics['roc_auc_ovr']:.4f}")
    
    print("\nClassification Report:")
    print(get_classification_report(
        metrics['predictions'],
        metrics['labels'],
        config['data']['class_names']
    ))
    
    # Save visualizations
    results_dir = Path(config['evaluation']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        config['data']['class_names'],
        save_path=str(results_dir / 'confusion_matrix.png')
    )
    
    roc_data = compute_roc_curves(
        metrics['probabilities'],
        metrics['labels'],
        config['data']['num_classes']
    )
    plot_roc_curves(
        roc_data,
        config['data']['class_names'],
        save_path=str(results_dir / 'roc_curves.png')
    )
    
    from src.evaluation.visualization import plot_training_history
    plot_training_history(history, save_path=str(results_dir / 'training_history.png'))
    
    print(f"\nResults saved to: {results_dir}")
    print("Training complete!")


if __name__ == '__main__':
    main()
