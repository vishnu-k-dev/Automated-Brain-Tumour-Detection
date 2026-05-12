"""
Brain Tumor Detection Web Application
Upload an MRI image to get instant classification with explainability.
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image
import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.preprocessing import preprocess_image
from src.data.augmentation import get_val_transforms
from src.models import create_model
from src.explainability.gradcam import GradCAM, apply_gradcam

# Global variables
model = None
device = None
config = None
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

def load_model():
    """Load the trained model."""
    global model, device, config
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create and load model
    model = create_model(config)
    checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    return model

def predict(image):
    """
    Classify brain MRI and generate Grad-CAM visualization.
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        Tuple of (predictions dict, gradcam image)
    """
    global model, device, config
    
    if model is None:
        load_model()
    
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert('RGB'))
    else:
        image_np = image
    
    original_image = image_np.copy()
    
    # Preprocess
    img_size = config['data']['img_size']
    processed = preprocess_image(
        image_np,
        size=(img_size, img_size),
        normalize=False,
        enhance=True
    )
    processed = (processed * 255).astype(np.uint8)
    
    # Transform
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
        
        probs = torch.softmax(logits, dim=-1)[0]
    
    # Create predictions dictionary
    predictions = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    
    # Generate Grad-CAM
    try:
        gradcam = GradCAM(model)
        heatmap = gradcam.generate(input_tensor)
        gradcam_image = apply_gradcam(original_image, heatmap, alpha=0.5)
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        gradcam_image = original_image
    
    return predictions, gradcam_image

def create_interface():
    """Create Gradio interface."""
    
    with gr.Blocks(title="Brain Tumor Detection", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🧠 Brain Tumor Detection & Classification
            
            Upload a brain MRI image to get instant classification using our 
            **Hybrid CNN-Vision Transformer** model with **99.3% accuracy**.
            
            **Supported Classes:** Glioma, Meningioma, No Tumor, Pituitary
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Brain MRI",
                    type="pil",
                    height=300
                )
                submit_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
                
                gr.Markdown(
                    """
                    ### Instructions:
                    1. Upload a brain MRI scan (JPG, PNG)
                    2. Click **Analyze**
                    3. View classification results and Grad-CAM
                    """
                )
            
            with gr.Column(scale=1):
                output_label = gr.Label(
                    label="Classification Results",
                    num_top_classes=4
                )
                output_gradcam = gr.Image(
                    label="Grad-CAM Visualization",
                    height=300
                )
        
        gr.Markdown(
            """
            ---
            ### About
            - **Model:** Hybrid CNN (ResNet50) + Vision Transformer
            - **Dataset:** Brain Tumor MRI Dataset (7,000+ images)
            - **Performance:** 99.31% Accuracy, 99.82% ROC-AUC
            - **Explainability:** Grad-CAM shows which regions influenced the prediction
            
            ⚠️ **Disclaimer:** This is a research tool, not for clinical diagnosis.
            """
        )
        
        # Connect components
        submit_btn.click(
            fn=predict,
            inputs=input_image,
            outputs=[output_label, output_gradcam]
        )
        
        # Example images
        gr.Examples(
            examples=[
                ["src/data/raw/Testing/glioma/Te-gl_0010.jpg"],
                ["src/data/raw/Testing/meningioma/Te-me_0010.jpg"],
                ["src/data/raw/Testing/notumor/Te-no_0010.jpg"],
                ["src/data/raw/Testing/pituitary/Te-pi_0010.jpg"],
            ],
            inputs=input_image,
            outputs=[output_label, output_gradcam],
            fn=predict,
            cache_examples=False
        )
    
    return demo

if __name__ == "__main__":
    print("Loading model...")
    load_model()
    
    print("Starting web interface...")
    demo = create_interface()
    demo.launch(
        share=False,  # Set to True to get a public link
        server_name="127.0.0.1",
        server_port=7860
    )
