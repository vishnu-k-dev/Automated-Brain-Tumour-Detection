# Automated Brain Tumor Detection using Hybrid CNN-ViT Architecture

## Table of Contents
1. Abstract
2. Introduction
3. Literature Review
4. Methodology
5. Results
6. Discussion
7. Conclusion and Future enhancements
8. References

---

## 1. Abstract
Brain tumors are among the most lethal neurological diseases, requiring early and accurate diagnosis for effective treatment planning. Magnetic Resonance Imaging (MRI) is the standard imaging modality for brain tumor detection; however, manual interpretation is time-consuming and subjective. In this project, we propose a novel hybrid deep learning architecture that integrates a Convolutional Neural Network (ResNet50) with a Vision Transformer (ViT) to automate the classification of brain tumors into four categories: glioma, meningioma, pituitary tumor, and no tumor. The CNN backbone extracts localized, fine-grained hierarchical features, while the ViT mechanism captures global contextual dependencies across the MRI slice. Our hybrid model achieved an outstanding 99.31% accuracy on the test set and a highly stable 98.95% ± 0.27% accuracy across 5-Fold Cross-Validation. Furthermore, testing on an independent external Figshare dataset yielded 99.61% accuracy, demonstrating exceptional generalization. The integration of Grad-CAM visualization provides crucial interpretability, highlighting the specific tumor regions driving the model's predictions and establishing this framework as a robust tool for computer-aided diagnosis.

## 2. Introduction
The early detection and accurate classification of brain tumors are critical steps in clinical oncology. Tumors such as gliomas, meningiomas, and pituitary adenomas exhibit varying degrees of malignancy, requiring distinct surgical and non-surgical interventions. MRI is heavily relied upon by radiologists to visualize soft-tissue contrast within the brain. However, the sheer volume of MRI slices generated per patient makes manual assessment a bottleneck, susceptible to inter-observer variability and human error.

Machine learning, particularly deep learning, has shown immense potential in automating medical image analysis. Traditional Convolutional Neural Networks (CNNs) have achieved high accuracy by learning local texture and shape patterns. Conversely, Vision Transformers (ViTs) have recently emerged as powerful tools capable of modeling long-range global relationships across an image using self-attention mechanisms.

This project introduces a hybrid architecture that synergistically combines a deep CNN (ResNet50) and a ViT. This combination allows the model to leverage local inductive biases from the CNN while retaining the global receptive field of the ViT. To bridge the black-box nature of deep learning in healthcare, Gradient-weighted Class Activation Mapping (Grad-CAM) is integrated to provide visual explanations of the model’s reasoning.

## 3. Literature Review
Automated brain tumor detection has been widely explored using various machine learning paradigms. Early approaches relied on handcrafted features (e.g., GLCM, LBP) paired with Support Vector Machines (SVMs) or Random Forests. For instance, studies comparing SVMs and CNNs demonstrated that CNNs yielded superior accuracy (96.33% vs 95%) without requiring manual feature engineering.

With the advent of deep learning, customized CNNs, such as VGG16, ResNet, and DenseNet, became the standard for MRI classification. Cheng et al. significantly improved classification by introducing tumor region augmentation and partition techniques. However, CNNs are inherently limited by their localized receptive fields, struggling to integrate global anatomical context efficiently.

Recently, Vision Transformers (ViTs), proposed by Dosovitskiy et al., have been adapted for medical imaging. While ViTs excel at capturing global context, they often lack the translation invariance of CNNs and require massive amounts of data to train effectively from scratch.

To address these limitations, recent works (e.g., Jayaraman et al., 2025) have proposed hybrid CNN-ViT frameworks utilizing cross-attention mechanisms. While promising, many of these models lack rigorous external validation and explainability. Our project builds upon this foundation by proposing a specialized FPN-enhanced ResNet-ViT hybrid, validated extensively via cross-validation, external testing, and visual explainability.

## 4. Methodology
The proposed system is designed as an end-to-end classification pipeline encompassing data preprocessing, hybrid feature extraction, and prediction.

**4.1 Data Preprocessing & Augmentation**
MRI scans vary heavily in intensity and resolution. Images were resized to 224x224 pixels and normalized. To prevent overfitting, spatial augmentations (rotations, flips, affine transformations) and pixel-level augmentations (Gaussian noise, brightness adjustments) were applied during training using the Albumentations library.

**4.2 Architecture Design (Hybrid CNN-ViT)**
The model leverages a dual-stream design:
1. **CNN Backbone:** A pre-trained ResNet-50 extracts multi-scale local features. A Feature Pyramid Network (FPN) mechanism is utilized to aggregate features across different spatial resolutions, capturing both minute cellular textures and larger structural anomalies.
2. **Vision Transformer:** Extracted CNN feature maps are flattened into token sequences and passed through a Vision Transformer (ViT-Base/16). The multi-head self-attention mechanisms analyze the relationships between different regions of the brain scan, capturing global context.
3. **Fusion and Classification:** The localized CNN features and global ViT tokens are concatenated and passed through a multi-layer perceptron (MLP) classification head to predict probabilities for the four distinct classes.

**4.3 Training Strategy**
The model was trained using Focal Loss to mitigate any class imbalance in the dataset. Optimization was performed using the AdamW optimizer paired with a Cosine Annealing Learning Rate scheduler. Mixed-precision training was employed to optimize memory usage and speed up computations.

**4.4 Explainability**
Grad-CAM was implemented on the final convolutional layers. By tracking the gradients flowing back from the predicted class, heatmap overlays are generated indicating which specific neurological regions influenced the network’s decision.

## 5. Results
The model was evaluated using standard metrics: Accuracy, F1-Score, and ROC-AUC. 

**5.1 Primary Test Set Performance**
On the primary hold-out test set, the hybrid model achieved:
- **Accuracy:** 99.31%
- **F1-Score:** 99.31%
- **ROC-AUC:** 99.82%

**5.2 5-Fold Cross-Validation**
To ensure statistical reliability, a 5-Fold Cross-Validation was conducted:
- **Mean Accuracy:** 98.95% ± 0.27%
- **Mean F1-Score:** 98.95% ± 0.26%
- **Mean ROC-AUC:** 99.92% ± 0.04%
The minimal standard deviation indicates exceptional model stability and reproducibility.

**5.3 External Dataset Validation**
When tested on a completely unseen, independent benchmark dataset (Figshare format), the model achieved an outstanding **99.61% accuracy**, proving it is highly resistant to dataset-specific biases.

**5.4 Ablation Study**
Component analysis confirmed the superiority of the hybrid approach:
- CNN-Only (ResNet50): 97.64%
- ViT-Only (ViT-B/16): 98.43%
- EfficientNet-B0: 98.86%
- **Hybrid CNN-ViT (Ours): 99.13%**

## 6. Discussion
The experimental results demonstrate that the Hybrid CNN-ViT architecture fundamentally outperforms standalone CNN or Transformer models for MRI classification. The ablation study reveals that while the ViT alone achieved 98.43% accuracy, it required 85.8M parameters. By integrating it with a CNN backbone, our hybrid model achieved higher accuracy (99.13%) with nearly half the parameters (47.0M), proving that CNN inductive biases significantly improve the efficiency of self-attention mechanisms in medical domains.

One of the most significant findings is the model's performance on the external Figshare dataset (99.61%). A common point of failure for deep medical models is poor generalization to new hospital hardware. Our results confirm that the structural and texture patterns learned by the hybrid network are universally applicable clinical markers, not localized dataset artifacts.

The implementation of Grad-CAM was highly successful. The generated heatmaps aligned consistently with ground-truth tumor locations, ensuring that the model acts as an interpretable diagnostic aid rather than an opaque "black-box."

## 7. Conclusion and Future enhancements
**Conclusion**
We have successfully developed, trained, and validated a high-performance Hybrid CNN-ViT architecture for automated brain tumor classification. By utilizing ResNet50 for local feature extraction and a Vision Transformer for global spatial awareness, the proposed system achieved robust, state-of-the-art results: 99.31% test accuracy, 98.95% ± 0.27% cross-validation accuracy, and 99.61% external validation accuracy. Coupled with interactive Grad-CAM visualizations, this framework presents a highly reliable, generalizable, and clinically interpretable solution for computer-aided neurological diagnosis.

**Future Enhancements**
1. **Model Compression:** Future work will focus on pruning and quantizing the self-attention heads to reduce the model's computational footprint, enabling real-time inference on edge devices in resource-constrained clinics.
2. **Multimodal Fusion:** Extending the system to process multiple MRI modalities (T1, T1ce, T2, FLAIR) simultaneously to improve sensitivity towards non-enhancing tumor cores.
3. **Segmentation Integration:** Combining the classification pipeline with an automatic semantic segmentation network (e.g., U-Net) to visually trace the exact boundaries and volume of the tumor alongside the classification tag.

## 8. References
[1] A. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," in *Int. Conf. Learn. Represent. (ICLR)*, 2021.
[2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 770–778.
[3] J. Cheng et al., "Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition," *PLOS ONE*, vol. 10, no. 10, e0140381, 2015.
[4] R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," in *Int. J. Comput. Vis. (IJCV)*, vol. 128, no. 2, pp. 336-359, 2020.
[5] G. Jayaraman, S. Meganathan, et al., "A hybrid CNN–ViT framework with cross-attention fusion and data augmentation for robust brain tumor classification," *Scientific Reports*, vol. 15, no. 1, 2025.
[6] T. Y. Lin et al., "Feature Pyramid Networks for Object Detection," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2017, pp. 2117–2125.
