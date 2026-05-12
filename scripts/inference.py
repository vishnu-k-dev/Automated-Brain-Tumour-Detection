"""
Inference script for single image prediction with explainability.
"""

import argparse
import yaml
import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import preprocess_image
from src.data.augmentation import get_val_transforms
from src.models import create_model
from src.explainability import GradCAM, apply_gradcam, visualize_attention


def main():
    parser = argparse.ArgumentParser(description='Brain Tumor Classifier Inference')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input MRI image')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output visualization')
    parser.add_argument('--show-gradcam', action='store_true',
                        help='Show Grad-CAM visualization')
    parser.add_argument('--show-attention', action='store_true',
                        help='Show ViT attention visualization')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names = config['data']['class_names']
    img_size = config['data']['img_size']
    
    # Load model
    print("Loading model...")
    model = create_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    print(f"Processing image: {args.image}")
    original_image = np.array(Image.open(args.image).convert('RGB'))
    
    # Preprocess
    processed = preprocess_image(
        original_image,
        size=(img_size, img_size),
        normalize=False,
        enhance=True
    )
    processed = (processed * 255).astype(np.uint8)
    
    # Apply validation transforms
    transform = get_val_transforms(img_size)
    transformed = transform(image=processed)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, dict):
            logits = output['logits']
        else:
            logits = output
        
        probs = torch.softmax(logits, dim=-1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()
    
    # Print results
    print("\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    print(f"\nPredicted Class: {class_names[pred_idx]}")
    print(f"Confidence: {confidence:.2%}")
    print("\nClass Probabilities:")
    for i, name in enumerate(class_names):
        prob = probs[0, i].item()
        bar = "█" * int(prob * 30)
        print(f"  {name:15} {prob:6.2%} {bar}")
    
    # Grad-CAM visualization
    if args.show_gradcam:
        print("\nGenerating Grad-CAM visualization...")
        from src.explainability.gradcam import visualize_gradcam
        
        save_path = args.output if args.output else None
        if save_path and not save_path.endswith('_gradcam.png'):
            save_path = save_path.replace('.png', '_gradcam.png')
        
        visualize_gradcam(
            model, input_tensor, original_image, class_names,
            save_path=save_path
        )
    
    # Attention visualization
    if args.show_attention:
        print("\nGenerating attention visualization...")
        from src.explainability.attention_viz import visualize_attention
        
        save_path = args.output if args.output else None
        if save_path and not save_path.endswith('_attention.png'):
            save_path = save_path.replace('.png', '_attention.png')
        
        visualize_attention(
            model, input_tensor, original_image, class_names,
            save_path=save_path
        )
    
    print("\nInference complete!")
    return pred_idx, confidence, probs[0].cpu().numpy()


if __name__ == '__main__':
    main()
