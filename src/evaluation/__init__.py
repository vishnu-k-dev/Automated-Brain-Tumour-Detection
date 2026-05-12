# Evaluation module initialization
from .metrics import evaluate_model, get_classification_report
from .visualization import plot_confusion_matrix, plot_roc_curves

__all__ = [
    'evaluate_model',
    'get_classification_report',
    'plot_confusion_matrix',
    'plot_roc_curves'
]
