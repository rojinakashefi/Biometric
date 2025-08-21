# 🧬 Biometric Recognition System

This repository implements a biometric recognition system that integrates fingerprint, iris, and facial recognition techniques. It combines classical image processing, statistical analysis, and deep learning to deliver robust identification and verification performance. 

## 📊 System Evaluation

The biometric systems were evaluated using a variety of performance metrics to assess their accuracy, reliability, and robustness:

- **Score Distributions**: Genuine vs. impostor score separation.

- **ROC & DET Curves**: True/False Positive and Negative Rates across thresholds.

- **F1 Score & Accuracy**: Evaluated at optimal thresholds for balanced performance.

- **Equal Error Rate (EER)**: The point where FAR equals FRR.

- **Area Under Curve (AUC)**: For both ROC and Precision-Recall curves.

- **Average Precision (AP)**: Weighted precision across recall levels.

- **CMC Curve**: Rank-based identification accuracy.

- **d-prime (d′)**: Statistical separation between genuine and impostor distributions.

- **Weighted Error Rate (WER)**: Custom error metric balancing security and convenience.

## 🔍 Fingerprint Recognition

This module focuses on extracting and matching fingerprint features using both classical and deep learning approaches:

- **Minutiae Detection**: Skeletonization and 3×3 pixel neighborhood analysis to identify ridge endings and bifurcations.

- **Orientation Matching**: Improves robustness to rotation, scale, and distortion.

- **Global Matching**: RANSAC-based alignment and Euclidean distance scoring.

- **Triplet Network**: Learns fingerprint embeddings for similarity comparison.

- **Texture Features**: Local Binary Patterns (LBP) for non-minutiae-based matching.

- **Score Fusion**: Combines fingerprint and iris scores for enhanced accuracy.

## 👁️ Iris Recognition

The iris recognition module enhances and compares iris images using segmentation and deep feature extraction:

- **Preprocessing**: Hough Transform for segmentation and Daugman Normalization for unwrapping.

- **Occlusion Handling**: Addresses eyelid, eyelash, and lighting interference.

- **Similarity Metrics**: Euclidean distance between embeddings; thresholding via Knee Locator.

- **t-SNE Visualization**: Demonstrates strong intra-class clustering and inter-class separation.

- **Score Fusion**: Weighted averaging with fingerprint scores to improve identification.

## 🧠 Facial Recognition

Facial recognition is implemented using a combination of statistical and deep learning methods:

- **Feature Extraction**:
  
  - PCA: Unsupervised dimensionality reduction.
  
  - LDA: Supervised projection maximizing class separability.
  
  - LBP: Texture-based descriptors with Chi-squared distance.
  
  - Deep Learning: Siamese and Triplet networks trained with contrastive/triplet loss.

- **Evaluation**:
  
  - Genuine vs. impostor score distributions.
  
  - F1 Score, Accuracy, EER, ROC, PR, and CMC curves.

- **Hyperparameter Tuning**:
  
  - PCA: Optimal components for F1 and Rank-1 accuracy.
  
  - LDA: Maximum class-based components.
  
  - LBP: Radius tuning for texture sensitivity.
  
  - Deep Learning: Embedding dimension optimization.
  
  ----
  
  This project is © 2025 KU Leuven and may not be used without permission.
