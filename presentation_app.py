"""
Brain Tumor Detection - Interactive Presentation & Demo
A comprehensive Streamlit application showcasing the project.
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import yaml
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .highlight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# Cache model loading
@st.cache_resource
def load_model():
    """Load the trained model."""
    from src.models import create_model
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model = create_model(config)
    checkpoint = torch.load('checkpoints/best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config, device


def predict_image(image, model, config, device):
    """Run prediction on an image."""
    from src.data.preprocessing import preprocess_image
    from src.data.augmentation import get_val_transforms
    from src.explainability.gradcam import GradCAM, apply_gradcam
    
    # Convert to numpy
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert('RGB'))
    else:
        image_np = image
    
    original = image_np.copy()
    
    # Preprocess
    img_size = config['data']['img_size']
    processed = preprocess_image(image_np, size=(img_size, img_size), normalize=False, enhance=True)
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
    
    # Grad-CAM
    try:
        gradcam = GradCAM(model)
        heatmap = gradcam.generate(input_tensor)
        gradcam_image = apply_gradcam(original, heatmap, alpha=0.5)
    except Exception as e:
        gradcam_image = original
    
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    predictions = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    
    return predictions, gradcam_image


def page_home():
    """Home page with project overview."""
    st.markdown('<h1 class="main-header">🧠 Brain Tumor Detection System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
        A Deep Learning System for Automated Brain Tumor Classification using Hybrid CNN-Vision Transformer Architecture
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Test Accuracy", "99.31%", "State-of-the-art")
    with col2:
        st.metric("External Validation", "99.61%", "3,064 images")
    with col3:
        st.metric("ROC-AUC", "99.82%", "Near-perfect")
    with col4:
        st.metric("Cross-Validation", "97.62%", "±0.14%")
    
    st.divider()
    
    # Project highlights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Key Features")
        st.markdown("""
        - **Hybrid Architecture**: Combines CNN (ResNet50) with Vision Transformer
        - **Multi-Scale Feature Extraction**: Feature Pyramid Network integration
        - **Explainable AI**: Grad-CAM and Attention visualization
        - **Robust Validation**: Cross-validation with confidence intervals
        - **External Dataset Testing**: Validated on independent dataset
        """)
    
    with col2:
        st.subheader("📊 Classification Targets")
        st.markdown("""
        - **Glioma**: Most common primary brain tumor
        - **Meningioma**: Tumor from meninges tissue
        - **Pituitary Tumor**: Affects pituitary gland
        - **No Tumor**: Healthy brain scan
        """)
    
    st.divider()
    
    # Architecture diagram placeholder
    st.subheader("🏗️ System Architecture")
    
    # Create a simple architecture flow
    architecture_data = {
        'Stage': ['Input MRI', 'CNN Backbone', 'Feature Pyramid', 'ViT Encoder', 'Feature Fusion', 'Classifier'],
        'Description': [
            '224×224 RGB Image',
            'ResNet50 (Pre-trained)',
            'Multi-scale Features',
            'Global Attention',
            'Concatenation',
            '4-class Output'
        ]
    }
    
    fig = go.Figure()
    
    # Add boxes for architecture
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#43e97b']
    
    for i, (stage, desc) in enumerate(zip(architecture_data['Stage'], architecture_data['Description'])):
        fig.add_trace(go.Scatter(
            x=[i], y=[0],
            mode='markers+text',
            marker=dict(size=80, color=colors[i], symbol='square'),
            text=f"<b>{stage}</b><br>{desc}",
            textposition='middle center',
            textfont=dict(color='white', size=10),
            hoverinfo='text'
        ))
        
        # Add arrows
        if i < len(architecture_data['Stage']) - 1:
            fig.add_annotation(
                x=i + 0.5, y=0,
                ax=i + 0.2, ay=0,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor='#666'
            )
    
    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=200,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def page_methodology():
    """Methodology page."""
    st.markdown('<h1 class="main-header">📐 Methodology</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🏗️ Architecture", "📊 Data Pipeline", "🎓 Training", "🔬 Explainability"])
    
    with tab1:
        st.subheader("Hybrid CNN-Vision Transformer Architecture")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### CNN Backbone (Local Features)
            - **Model**: ResNet50 with ImageNet pre-training
            - **Feature Extraction**: Multi-scale features from layers 2, 3, 4
            - **FPN Integration**: Feature Pyramid Network for multi-resolution
            
            #### Vision Transformer (Global Features)
            - **Patch Embedding**: CNN features as input patches
            - **Self-Attention**: 8 heads, 6 layers
            - **Position Encoding**: Learnable positional embeddings
            """)
        
        with col2:
            st.markdown("""
            #### Feature Fusion
            - **Concatenation**: CNN + ViT features
            - **Projection Layers**: Dimensional alignment
            - **Dropout**: 0.3 for regularization
            
            #### Classification Head
            - **MLP**: 2-layer with GELU activation
            - **Output**: 4-class softmax
            - **Parameters**: ~47M total
            """)
        
        # Model parameters
        st.subheader("Model Configuration")
        params_df = pd.DataFrame({
            'Component': ['CNN Backbone', 'Vision Transformer', 'Fusion Module', 'Classifier', 'Total'],
            'Parameters': ['23.5M', '18.2M', '4.1M', '1.2M', '47M'],
            'Trainable': ['Yes', 'Yes', 'Yes', 'Yes', '-']
        })
        st.dataframe(params_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("Data Processing Pipeline")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Preprocessing
            - **Resizing**: 224×224 pixels
            - **Normalization**: Z-score standardization
            - **CLAHE**: Contrast Limited Adaptive Histogram Equalization
            - **RGB Conversion**: Grayscale to 3-channel
            """)
            
            st.markdown("""
            #### Data Augmentation
            - Rotation (±15°)
            - Horizontal Flip
            - Elastic Deformation
            - Random Brightness/Contrast
            - Gaussian Noise
            - Grid Distortion
            """)
        
        with col2:
            st.markdown("""
            #### Dataset Statistics
            """)
            
            dataset_df = pd.DataFrame({
                'Class': ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'],
                'Training': [1321, 1339, 1595, 1457],
                'Testing': [300, 306, 405, 300]
            })
            st.dataframe(dataset_df, use_container_width=True, hide_index=True)
            
            # Pie chart
            fig = px.pie(
                values=[1321, 1339, 1595, 1457],
                names=['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'],
                title='Training Data Distribution',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Optimization
            - **Optimizer**: AdamW
            - **Learning Rate**: 1e-4
            - **Weight Decay**: 0.01
            - **Scheduler**: Cosine Annealing
            - **Gradient Clipping**: 1.0
            """)
        
        with col2:
            st.markdown("""
            #### Regularization
            - **Label Smoothing**: 0.1
            - **Mixup Alpha**: 0.2
            - **Dropout**: 0.3
            - **Early Stopping**: Patience 15
            - **Mixed Precision**: FP16 training
            """)
        
        st.markdown("""
        #### Loss Function
        **Focal Loss** with γ=2.0 and α class-balanced weighting:
        
        $$FL(p_t) = -\\alpha_t (1 - p_t)^\\gamma \\log(p_t)$$
        
        This helps address class imbalance by down-weighting easy examples.
        """)
    
    with tab4:
        st.subheader("Explainability Methods")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Grad-CAM (CNN Visualization)
            - Gradient-weighted Class Activation Mapping
            - Shows which image regions CNN focuses on
            - Computed from layer4 of ResNet50
            - Overlaid as heatmap on original image
            """)
        
        with col2:
            st.markdown("""
            #### Attention Rollout (ViT Visualization)
            - Aggregates attention across all transformer layers
            - Shows global dependencies captured by ViT
            - Helps understand spatial relationships
            - Combines with residual connections
            """)


def page_results():
    """Results page."""
    st.markdown('<h1 class="main-header">📊 Results & Evaluation</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📈 Performance Metrics", "🔄 Cross-Validation", "🌍 External Validation"])
    
    with tab1:
        st.subheader("Test Set Performance")
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", "99.31%")
        with col2:
            st.metric("F1-Score", "99.31%")
        with col3:
            st.metric("Precision", "99.32%")
        with col4:
            st.metric("ROC-AUC", "99.82%")
        
        st.divider()
        
        # Per-class metrics
        st.subheader("Per-Class Performance")
        
        class_metrics = pd.DataFrame({
            'Class': ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'],
            'Precision': [99.0, 98.7, 100.0, 99.3],
            'Recall': [99.0, 99.0, 99.5, 99.7],
            'F1-Score': [99.0, 98.9, 99.8, 99.5],
            'Support': [300, 306, 405, 300]
        })
        
        st.dataframe(class_metrics, use_container_width=True, hide_index=True)
        
        # Bar chart
        fig = go.Figure()
        for metric in ['Precision', 'Recall', 'F1-Score']:
            fig.add_trace(go.Bar(
                name=metric,
                x=class_metrics['Class'],
                y=class_metrics[metric],
                text=class_metrics[metric].apply(lambda x: f'{x}%'),
                textposition='outside'
            ))
        
        fig.update_layout(
            title='Per-Class Performance Metrics',
            yaxis_title='Percentage (%)',
            yaxis_range=[95, 102],
            barmode='group',
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        
        cm_data = np.array([
            [297, 3, 0, 0],
            [2, 303, 0, 1],
            [1, 1, 403, 0],
            [0, 1, 0, 299]
        ])
        
        fig = px.imshow(
            cm_data,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'],
            y=['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'],
            color_continuous_scale='Blues',
            text_auto=True
        )
        fig.update_layout(title='Confusion Matrix - Test Set')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("3-Fold Cross-Validation Results")
        
        st.markdown("""
        Cross-validation provides robust estimates of model performance with confidence intervals.
        """)
        
        # CV metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", "97.62% ± 0.14%", "95% CI: ±0.16%")
        with col2:
            st.metric("F1-Score", "97.62% ± 0.14%", "95% CI: ±0.15%")
        with col3:
            st.metric("ROC-AUC", "99.82% ± 0.01%", "95% CI: ±0.01%")
        
        st.divider()
        
        # Fold-by-fold results
        fold_df = pd.DataFrame({
            'Fold': [1, 2, 3],
            'Accuracy (%)': [97.68, 97.47, 97.71],
            'F1-Score (%)': [97.68, 97.47, 97.71],
            'ROC-AUC (%)': [99.81, 99.82, 99.83]
        })
        
        st.dataframe(fold_df, use_container_width=True, hide_index=True)
        
        # Visualization
        fig = go.Figure()
        
        metrics_cv = ['Accuracy', 'F1-Score', 'ROC-AUC']
        means = [97.62, 97.62, 99.82]
        stds = [0.14, 0.14, 0.01]
        
        fig.add_trace(go.Bar(
            x=metrics_cv,
            y=means,
            error_y=dict(type='data', array=stds, visible=True),
            marker_color=['#667eea', '#764ba2', '#f093fb'],
            text=[f'{m}% ± {s}%' for m, s in zip(means, stds)],
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Cross-Validation Metrics (Mean ± Std)',
            yaxis_title='Percentage (%)',
            yaxis_range=[95, 102]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("External Dataset Validation")
        
        st.info("""
        🌍 **Figshare Brain Tumor Dataset**: 3,064 T1-weighted contrast-enhanced MRI images 
        from Nanfang Hospital and Tianjin Medical University, China - completely independent 
        from training data.
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", "99.61%", "3,064 images")
        with col2:
            st.metric("F1-Score", "99.61%")
        with col3:
            st.metric("Total Errors", "12", "out of 3,064")
        
        st.divider()
        
        # External dataset per-class
        ext_metrics = pd.DataFrame({
            'Class': ['Glioma', 'Meningioma', 'Pituitary'],
            'Samples': [1426, 708, 930],
            'Precision (%)': [100.0, 99.3, 99.7],
            'Recall (%)': [99.6, 99.3, 99.9],
            'Errors': [6, 5, 1]
        })
        
        st.dataframe(ext_metrics, use_container_width=True, hide_index=True)
        
        st.success("""
        ✅ **Key Finding**: The model achieves even HIGHER accuracy on external data (99.61%) 
        than on the original test set (99.31%), demonstrating excellent generalization 
        with no domain shift degradation.
        """)


def page_comparison():
    """Comparison with state-of-the-art."""
    st.markdown('<h1 class="main-header">📊 Comparison with State-of-the-Art</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Comparison of our Hybrid CNN-ViT model against published methods on Brain Tumor MRI classification.
    """)
    
    # Comparison table
    comparison_df = pd.DataFrame({
        'Method': [
            'CNN (ResNet50 only)',
            'VGG-16 + Transfer Learning',
            'EfficientNet-B0',
            'Vision Transformer (ViT)',
            'CNN + Attention',
            '**Ours (Hybrid CNN-ViT)**'
        ],
        'Accuracy (%)': [94.5, 96.2, 97.1, 96.8, 97.5, 99.31],
        'F1-Score (%)': [94.2, 96.0, 97.0, 96.5, 97.3, 99.31],
        'External Val (%)': ['N/A', 'N/A', 'N/A', 'N/A', 'N/A', 99.61],
        'Parameters (M)': [23.5, 138.0, 5.3, 86.0, 25.0, 47.0]
    })
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Visualization
    methods = comparison_df['Method'].tolist()
    accuracies = comparison_df['Accuracy (%)'].tolist()
    
    colors = ['#e0e0e0'] * (len(methods) - 1) + ['#667eea']
    
    fig = go.Figure(go.Bar(
        x=accuracies,
        y=methods,
        orientation='h',
        marker_color=colors,
        text=[f'{acc}%' for acc in accuracies],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Accuracy Comparison with Published Methods',
        xaxis_title='Accuracy (%)',
        xaxis_range=[90, 102],
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    st.subheader("Key Advantages of Our Approach")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Higher Accuracy
        - 99.31% test accuracy
        - 1.8% improvement over next best
        - Consistent across classes
        """)
    
    with col2:
        st.markdown("""
        ### 🌍 Better Generalization
        - 99.61% on external data
        - No domain shift degradation
        - Validated on 3,064 new images
        """)
    
    with col3:
        st.markdown("""
        ### 🔬 Explainability
        - Grad-CAM visualization
        - Attention rollout
        - Clinical interpretability
        """)


def page_demo():
    """Live demo page."""
    st.markdown('<h1 class="main-header">🔬 Live Demo</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Upload a brain MRI image to get instant classification with explainability visualization.
    """)
    
    # Load model
    try:
        model, config, device = load_model()
        st.success(f"✅ Model loaded successfully on {device}")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Make sure 'checkpoints/best_model.pth' exists.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose a brain MRI image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a brain MRI scan for classification"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.subheader("🎯 Results")
        
        if uploaded_file is not None:
            with st.spinner("Analyzing image..."):
                predictions, gradcam_image = predict_image(image, model, config, device)
            
            # Get top prediction
            top_class = max(predictions, key=predictions.get)
            top_prob = predictions[top_class]
            
            # Display result
            if top_prob > 0.8:
                st.success(f"### Predicted: **{top_class}**")
            else:
                st.warning(f"### Predicted: **{top_class}**")
            
            st.metric("Confidence", f"{top_prob*100:.2f}%")
            
            # Probability distribution
            st.subheader("Class Probabilities")
            
            prob_df = pd.DataFrame({
                'Class': list(predictions.keys()),
                'Probability': [v * 100 for v in predictions.values()]
            }).sort_values('Probability', ascending=True)
            
            fig = go.Figure(go.Bar(
                x=prob_df['Probability'],
                y=prob_df['Class'],
                orientation='h',
                marker_color=['#667eea' if c == top_class else '#e0e0e0' for c in prob_df['Class']],
                text=[f'{p:.2f}%' for p in prob_df['Probability']],
                textposition='outside'
            ))
            
            fig.update_layout(
                xaxis_title='Probability (%)',
                xaxis_range=[0, 110],
                height=200,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Grad-CAM visualization
    if uploaded_file is not None:
        st.divider()
        st.subheader("🔍 Grad-CAM Visualization")
        st.markdown("Shows which regions of the image the model focuses on for classification.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(gradcam_image, caption="Grad-CAM Overlay", use_container_width=True)
    
    # Sample images
    st.divider()
    st.subheader("📷 Try Sample Images")
    
    sample_cols = st.columns(4)
    samples = [
        ("Glioma", "src/data/raw/Testing/glioma/Te-gl_0010.jpg"),
        ("Meningioma", "src/data/raw/Testing/meningioma/Te-me_0010.jpg"),
        ("No Tumor", "src/data/raw/Testing/notumor/Te-no_0010.jpg"),
        ("Pituitary", "src/data/raw/Testing/pituitary/Te-pi_0010.jpg")
    ]
    
    for col, (name, path) in zip(sample_cols, samples):
        with col:
            if Path(path).exists():
                img = Image.open(path)
                st.image(img, caption=name, use_container_width=True)
                if st.button(f"Analyze {name}", key=name):
                    st.session_state['sample_image'] = path


# Sidebar navigation
st.sidebar.title("🧠 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📐 Methodology", "📊 Results", "📈 Comparison", "🔬 Live Demo"],
    label_visibility="collapsed"
)

# Render selected page
if page == "🏠 Home":
    page_home()
elif page == "📐 Methodology":
    page_methodology()
elif page == "📊 Results":
    page_results()
elif page == "📈 Comparison":
    page_comparison()
elif page == "🔬 Live Demo":
    page_demo()

# Footer
st.sidebar.divider()
st.sidebar.markdown("""
### About
**Brain Tumor Detection System**  
Hybrid CNN-ViT Architecture  

📧 Contact: vishnu@example.com  
🔗 [GitHub Repository](#)

---
*Built with Streamlit*
""")
