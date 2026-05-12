"""
Brain Tumor Detection & Classification — Hugging Face Spaces Demo
Hybrid CNN-Vision Transformer with Grad-CAM Explainability

Upload a brain MRI image to get instant classification with Grad-CAM visualization.
"""

import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import yaml
import sys
import os
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data.preprocessing import preprocess_image
from src.data.augmentation import get_val_transforms
from src.models import create_model
from src.explainability.gradcam import GradCAM, apply_gradcam

# ─── Constants ───────────────────────────────────────────────────────────────
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
CLASS_DESCRIPTIONS = {
    'Glioma': 'Most common primary brain tumor, arising from glial cells',
    'Meningioma': 'Tumor arising from the meninges surrounding the brain',
    'No Tumor': 'Healthy brain MRI with no detectable tumor',
    'Pituitary': 'Tumor in the pituitary gland at the base of the brain',
}

# ─── Global State ────────────────────────────────────────────────────────────
model = None
device = None
config = None


def load_model():
    """Load the trained model (CPU-only for HF Spaces free tier)."""
    global model, device, config

    device = torch.device('cpu')  # HF free tier = CPU only

    # Load config
    config_path = ROOT / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create model
    model = create_model(config)

    # Load checkpoint
    checkpoint_path = ROOT / 'checkpoints' / 'best_model.pth'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ Model loaded from {checkpoint_path}")
    else:
        print(f"⚠️ No checkpoint found at {checkpoint_path} — using random weights")

    model = model.to(device)
    model.eval()
    return model


def predict(image):
    """
    Classify brain MRI and generate Grad-CAM visualization.

    Args:
        image: PIL Image from Gradio

    Returns:
        Tuple of (predictions dict, gradcam image)
    """
    global model, device, config

    if model is None:
        load_model()

    if image is None:
        return {name: 0.0 for name in CLASS_NAMES}, None

    # Convert to numpy array
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
    predictions = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

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
    """Create the Gradio interface for HF Spaces."""

    # Custom CSS for a polished look
    custom_css = """
    .gradio-container {
        max-width: 960px !important;
        margin: auto !important;
    }
    .header-text {
        text-align: center;
        margin-bottom: 0.5em;
    }
    .disclaimer {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 12px;
        margin-top: 8px;
        font-size: 0.9em;
        color: #856404;
    }
    .metrics-table {
        font-size: 0.95em;
    }
    """

    with gr.Blocks(
        title="🧠 Brain Tumor Detection",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
        ),
        css=custom_css
    ) as demo:

        # ── Header ──────────────────────────────────────────────
        gr.Markdown(
            """
            <div class="header-text">

            # 🧠 Brain Tumor Detection & Classification

            Upload a brain MRI image to get instant classification using our
            **Hybrid CNN-Vision Transformer** model with **Grad-CAM explainability**.

            **4 Classes:** Glioma · Meningioma · No Tumor · Pituitary

            </div>
            """
        )

        # ── Main Content ────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="📤 Upload Brain MRI",
                    type="pil",
                    height=320,
                    sources=["upload", "clipboard"]
                )
                submit_btn = gr.Button(
                    "🔍 Analyze MRI",
                    variant="primary",
                    size="lg"
                )

                gr.Markdown(
                    """
                    ### 📋 Instructions
                    1. Upload a brain MRI scan (JPG, PNG)
                    2. Click **Analyze MRI**
                    3. View classification + Grad-CAM heatmap

                    Grad-CAM highlights the regions that most
                    influenced the model's prediction.
                    """
                )

            with gr.Column(scale=1):
                output_label = gr.Label(
                    label="📊 Classification Results",
                    num_top_classes=4
                )
                output_gradcam = gr.Image(
                    label="🔥 Grad-CAM Visualization",
                    height=320
                )

        # ── Example Images ──────────────────────────────────────
        example_dir = ROOT / "examples"
        if example_dir.exists():
            example_images = sorted(example_dir.glob("*.jpg")) + sorted(example_dir.glob("*.png"))
            if example_images:
                gr.Examples(
                    examples=[[str(p)] for p in example_images[:4]],
                    inputs=input_image,
                    outputs=[output_label, output_gradcam],
                    fn=predict,
                    cache_examples=False
                )

        # ── Model Info ──────────────────────────────────────────
        with gr.Accordion("📈 Model Performance & Architecture", open=False):
            gr.Markdown(
                """
                | Metric | Score |
                |--------|-------|
                | **Accuracy** | 99.31% |
                | **F1-Score (Weighted)** | 99.30% |
                | **ROC-AUC** | 99.92% |
                | **5-Fold CV Accuracy** | 98.95% ± 0.27% |

                ---

                ### Architecture
                ```
                MRI Image (224×224)
                    → ResNet50 CNN Backbone (local features)
                    → Vision Transformer (6 layers, 8 heads — global context)
                    → Learnable Radiomics Branch (texture + shape)
                    → Feature Fusion (2048 + 512 + 128 dims)
                    → Classification Head → 4-Class Prediction
                ```

                ### Ablation Study
                | Model | Accuracy |
                |-------|----------|
                | CNN Only (ResNet50) | 97.64% |
                | ViT Only (ViT-B/16) | 98.43% |
                | EfficientNet-B0 | 98.86% |
                | **Hybrid CNN-ViT (Ours)** | **99.13%** |

                ### Dataset
                [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) —
                7,023 images across 4 classes.

                ### Training Details
                - **Optimizer:** AdamW (lr=1e-4, weight_decay=0.01)
                - **Scheduler:** Cosine annealing with 5-epoch warmup
                - **Augmentation:** Rotation, elastic deform, mixup, CLAHE enhancement
                - **Regularization:** Label smoothing (0.1), dropout (0.3), gradient clipping
                - **Parameters:** ~47M total
                """
            )

        # ── Disclaimer ──────────────────────────────────────────
        gr.Markdown(
            """
            <div class="disclaimer">
            ⚠️ <strong>Disclaimer:</strong> This is a research demonstration tool and is
            <strong>NOT intended for clinical diagnosis</strong>. Always consult qualified
            medical professionals for medical imaging interpretation.
            </div>
            """
        )

        # ── Connect ─────────────────────────────────────────────
        submit_btn.click(
            fn=predict,
            inputs=input_image,
            outputs=[output_label, output_gradcam]
        )

    return demo


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading model...")
    load_model()

    print("Starting Gradio interface...")
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
