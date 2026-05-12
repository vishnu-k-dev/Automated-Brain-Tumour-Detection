import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import yaml
import h5py
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import preprocess_image
from src.data.augmentation import get_val_transforms
from src.models import create_model


def load_mat_image(mat_path):
    """Load image and label from .mat file (MATLAB v7.3 HDF5 format)."""
    try:
        with h5py.File(mat_path, 'r') as f:
            cjdata = f['cjdata']
            
            # Extract image (transpose for correct orientation)
            image = cjdata['image'][()].T
            
            # Normalize to 0-255
            image = image.astype(np.float64)
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            image = (image * 255).astype(np.uint8)
            
            # Convert to RGB (grayscale to 3-channel)
            if len(image.shape) == 2:
                image = np.stack([image, image, image], axis=-1)
            
            # Extract label (1=meningioma, 2=glioma, 3=pituitary)
            label = int(cjdata['label'][()].flatten()[0])
            
            return image, label
    except Exception as e:
        print(f"Error loading {mat_path}: {e}")
        return None, None


def evaluate_external_dataset(
    data_dir: str,
    checkpoint_path: str,
    config_path: str = 'config/config.yaml'
):
    """Evaluate model on external dataset."""
    
    # Load config and model
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create and load model
    print("Loading model...")
    model = create_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Label mapping: External dataset -> Our model classes
    # External: 1=meningioma, 2=glioma, 3=pituitary
    # Our model: 0=glioma, 1=meningioma, 2=notumor, 3=pituitary
    ext_to_our = {
        1: 1,  # meningioma -> 1
        2: 0,  # glioma -> 0
        3: 3   # pituitary -> 3
    }
    
    ext_class_names = {1: 'meningioma', 2: 'glioma', 3: 'pituitary'}
    our_class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    # Transforms
    img_size = config['data']['img_size']
    transform = get_val_transforms(img_size)
    
    # Find all .mat files
    data_path = Path(data_dir)
    mat_files = []
    for subdir in data_path.iterdir():
        if subdir.is_dir():
            mat_files.extend(list(subdir.glob('*.mat')))
    
    print(f"Found {len(mat_files)} .mat files")
    
    # Evaluate
    all_preds = []
    all_labels = []
    all_probs = []
    errors = 0
    
    print("Evaluating...")
    for mat_file in tqdm(mat_files):
        image, ext_label = load_mat_image(str(mat_file))
        
        if image is None:
            errors += 1
            continue
        
        # Convert external label to our label
        our_label = ext_to_our.get(ext_label)
        if our_label is None:
            continue
        
        # Preprocess
        processed = preprocess_image(
            image,
            size=(img_size, img_size),
            normalize=False,
            enhance=True
        )
        processed = (processed * 255).astype(np.uint8)
        
        # Transform
        transformed = transform(image=processed)
        input_tensor = transformed['image'].unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            if isinstance(output, dict):
                logits = output['logits']
            else:
                logits = output
            
            probs = torch.softmax(logits, dim=-1)[0]
            pred = probs.argmax().item()
        
        all_preds.append(pred)
        all_labels.append(our_label)
        all_probs.append(probs.cpu().numpy())
    
    # Calculate metrics
    print(f"\n{'='*60}")
    print("EXTERNAL DATASET EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total samples evaluated: {len(all_labels)}")
    print(f"Loading errors: {errors}")
    
    # Filter to only 3 classes that exist in external dataset (exclude notumor)
    valid_classes = [0, 1, 3]  # glioma, meningioma, pituitary
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', labels=valid_classes)
    precision = precision_score(all_labels, all_preds, average='weighted', labels=valid_classes, zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', labels=valid_classes, zero_division=0)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    
    # Per-class report
    print(f"\nClassification Report:")
    print(classification_report(
        all_labels, all_preds,
        labels=valid_classes,
        target_names=[our_class_names[i] for i in valid_classes],
        zero_division=0
    ))
    
    # Confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds, labels=valid_classes)
    print(f"              {'  '.join([our_class_names[i][:8] for i in valid_classes])}")
    for i, row in enumerate(cm):
        print(f"{our_class_names[valid_classes[i]]:<12} {row}")
    
    # Class distribution
    print(f"\nTrue Label Distribution:")
    label_counts = Counter(all_labels)
    for label, count in sorted(label_counts.items()):
        print(f"  {our_class_names[label]}: {count}")
    
    print(f"\nPredicted Distribution:")
    pred_counts = Counter(all_preds)
    for pred, count in sorted(pred_counts.items()):
        print(f"  {our_class_names[pred]}: {count}")
    
    # Check if model is predicting "notumor" (class 2) incorrectly
    notumor_preds = pred_counts.get(2, 0)
    if notumor_preds > 0:
        print(f"\n⚠️  Warning: Model predicted 'notumor' {notumor_preds} times")
        print("   (This class doesn't exist in external dataset)")
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate on external dataset')
    parser.add_argument('--data-dir', type=str, 
                        default='src/data/ext_test',
                        help='Path to external dataset')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str,
                        default='config/config.yaml',
                        help='Path to config file')
    
    args = parser.parse_args()
    
    evaluate_external_dataset(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        config_path=args.config
    )
