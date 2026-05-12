"""
Generate missing visualizations from training results.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_data_loaders
from src.models import create_model
from src.evaluation import evaluate_model
from src.evaluation.visualization import plot_roc_curves, plot_training_history
from src.evaluation.metrics import compute_roc_curves
import yaml

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
print("Loading data...")
loaders = get_data_loaders(
    data_dir='src/data/raw',
    batch_size=16,
    img_size=224,
    num_workers=0
)

# Load model
print("Loading model...")
model = create_model(config)
checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Evaluate
print("Evaluating...")
class_names = config['data']['class_names']
metrics = evaluate_model(model, loaders['test'], device, class_names=class_names)

# Generate ROC curves
print("Generating ROC curves...")
roc_data = compute_roc_curves(
    metrics['probabilities'],
    metrics['labels'],
    config['data']['num_classes']
)

plt.figure(figsize=(10, 8))
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
plt.savefig('results/roc_curves.png', dpi=300, bbox_inches='tight')
print("Saved: results/roc_curves.png")
plt.close()

print("Done!")
