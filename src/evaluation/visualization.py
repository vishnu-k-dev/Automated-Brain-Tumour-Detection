"""
Visualization utilities for evaluation results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True,
    figsize: tuple = (10, 8)
):
    """Plot confusion matrix with styling."""
    plt.figure(figsize=figsize)
    
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_norm
        fmt = '.2%'
    else:
        cm_display = cm
        fmt = 'd'
    
    # Create heatmap
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return plt.gcf()


def plot_roc_curves(
    roc_data: Dict,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8)
):
    """Plot ROC curves for all classes."""
    plt.figure(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(class_names)))
    
    for i, class_name in enumerate(class_names):
        if i in roc_data:
            data = roc_data[i]
            plt.plot(
                data['fpr'], data['tpr'],
                color=colors[i],
                lw=2,
                label=f'{class_name} (AUC = {data["auc"]:.3f})'
            )
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return plt.gcf()


def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 5)
):
    """Plot training history."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    epochs = range(1, len(history['train']) + 1)
    
    # Loss
    train_loss = [h['loss'] for h in history['train']]
    val_loss = [h['loss'] for h in history['val']]
    
    axes[0].plot(epochs, train_loss, 'b-', label='Train')
    axes[0].plot(epochs, val_loss, 'r-', label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    train_acc = [h.get('accuracy', 0) for h in history['train']]
    val_acc = [h.get('accuracy', 0) for h in history['val']]
    
    axes[1].plot(epochs, train_acc, 'b-', label='Train')
    axes[1].plot(epochs, val_acc, 'r-', label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # F1 Score
    val_f1 = [h.get('f1', 0) for h in history['val']]
    
    axes[2].plot(epochs, val_f1, 'g-', label='Validation F1')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('Validation F1 Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_calibration_curve(
    calibration_data: Dict,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 8)
):
    """Plot reliability/calibration diagram."""
    plt.figure(figsize=figsize)
    
    bin_accuracies = calibration_data['bin_accuracies']
    bin_confidences = calibration_data['bin_confidences']
    
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    plt.bar(
        np.linspace(0.05, 0.95, len(bin_accuracies)),
        bin_accuracies,
        width=0.08,
        alpha=0.7,
        color='blue',
        label=f"Model (ECE = {calibration_data['ece']:.3f})"
    )
    
    plt.xlabel('Mean Predicted Confidence', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title('Calibration Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return plt.gcf()


def plot_class_distribution(
    labels: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
):
    """Plot class distribution."""
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(class_names)))
    
    bars = plt.bar(class_names, counts, color=colors)
    
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', fontsize=10)
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Class Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return plt.gcf()
