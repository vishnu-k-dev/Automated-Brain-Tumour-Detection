"""
Comprehensive evaluation metrics for brain tumor classification.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from scipy import stats
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str] = None,
    return_predictions: bool = True
) -> Dict:
    """
    Comprehensive model evaluation.
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs['logits']
            
            probs = torch.softmax(outputs, dim=-1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision_weighted': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall_weighted': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
    }
    
    # Per-class metrics
    num_classes = all_probs.shape[1]
    for i in range(num_classes):
        class_name = class_names[i] if class_names else str(i)
        binary_labels = (all_labels == i).astype(int)
        binary_preds = (all_preds == i).astype(int)
        
        metrics[f'precision_{class_name}'] = precision_score(binary_labels, binary_preds, zero_division=0)
        metrics[f'recall_{class_name}'] = recall_score(binary_labels, binary_preds, zero_division=0)
        metrics[f'f1_{class_name}'] = f1_score(binary_labels, binary_preds, zero_division=0)
    
    # ROC-AUC
    try:
        metrics['roc_auc_ovr'] = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        metrics['roc_auc_ovo'] = roc_auc_score(all_labels, all_probs, multi_class='ovo')
    except ValueError:
        metrics['roc_auc_ovr'] = 0.0
        metrics['roc_auc_ovo'] = 0.0
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(all_labels, all_preds)
    
    if return_predictions:
        metrics['predictions'] = all_preds
        metrics['labels'] = all_labels
        metrics['probabilities'] = all_probs
    
    return metrics


def get_classification_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: List[str] = None
) -> str:
    """Get detailed classification report."""
    return classification_report(
        labels, predictions,
        target_names=class_names,
        digits=4,
        zero_division=0
    )


def compute_calibration(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> Dict:
    """Compute calibration metrics."""
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = (predictions == labels)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if np.sum(in_bin) > 0:
            bin_accuracies.append(np.mean(accuracies[in_bin]))
            bin_confidences.append(np.mean(confidences[in_bin]))
            bin_counts.append(np.sum(in_bin))
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)
            bin_counts.append(0)
    
    # Expected Calibration Error
    ece = sum([count * abs(acc - conf) for acc, conf, count in 
               zip(bin_accuracies, bin_confidences, bin_counts)]) / len(labels)
    
    return {
        'ece': ece,
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts
    }


def mcnemar_test(
    preds1: np.ndarray,
    preds2: np.ndarray,
    labels: np.ndarray
) -> Dict:
    """
    McNemar's test for comparing two classifiers.
    """
    correct1 = (preds1 == labels)
    correct2 = (preds2 == labels)
    
    # Contingency table
    b = np.sum(correct1 & ~correct2)  # Model 1 correct, Model 2 wrong
    c = np.sum(~correct1 & correct2)  # Model 1 wrong, Model 2 correct
    
    # McNemar's test with continuity correction
    if b + c == 0:
        statistic = 0
        p_value = 1.0
    else:
        statistic = ((abs(b - c) - 1) ** 2) / (b + c)
        p_value = 1 - stats.chi2.cdf(statistic, 1)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'b': b,
        'c': c,
        'significant': p_value < 0.05
    }


def compute_roc_curves(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_classes: int
) -> Dict:
    """Compute ROC curves for each class."""
    roc_data = {}
    
    for i in range(num_classes):
        binary_labels = (labels == i).astype(int)
        fpr, tpr, thresholds = roc_curve(binary_labels, probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        
        roc_data[i] = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc
        }
    
    return roc_data
