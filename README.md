# CWRU Bearing Fault Classification with 1D Convolutional Neural Networks

**Author:** Spencer Kenison Deep Learning Theory & Applications - Final Project

*Note: This project was developed as part of STAT 6685 at Utah State University (USU).*

This repository contains a deep learning framework for the classification of mechanical bearing faults using the Case Western Reserve University (CWRU) Bearing Dataset. The project implements a 1D Convolutional Neural Network (CNN) in PyTorch to classify vibration signals into four categories: Normal, Inner Race Fault (IR), Outer Race Fault (OR), and Ball Fault.

## Features
- **Data Pipeline**: Automatic download and caching of the CWRU dataset via kagglehub.
- **Signal Processing**: Multi-channel (Drive End & Fan End) signal segmentation using a sliding window approach (1024 samples).
- **Deep Learning Model**: A 1D-CNN LSTM MLP architecture designed for time-series vibration analysis.
- **Explainability**: Implementation of 1D Grad-CAM to visualize which parts of the vibration signal the model focuses on for specific fault detections.
- **Evaluation**: Comprehensive metrics including Accuracy, Confusion Matrix, and Classification Reports.

## Requirements
To run this notebook, you will need the following libraries:
- torch (PyTorch)
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- seaborn
- kagglehub
- tqdm

## Dataset Structure
The code utilizes the CWRU Bearing Dataset. It maps .mat files to fault classes based on filename patterns:
- normal -> Class 0
- ir (Inner Raceway) -> Class 1
- or (Outer Raceway) -> Class 2
- b (Ball) -> Class 3

Data is segmented into discrete chunks of 1024 data points and normalized using Z-score standardization.

## Model Architecture
The project utilizes a hybrid deep learning architecture that combines spatial feature extraction with temporal sequence modeling to achieve robust fault classification.

The model consists of three primary stages:

1.  **1D-Convolutional Neural Network (CNN) Stage**:
    * **Feature Extraction**: Multiple 1D-convolutional layers use sliding kernels to capture local spatial patterns and transient features in the vibration signals.
    * **Batch Normalization**: Applied after each convolution to accelerate training and provide a regularization effect.
    * **Max Pooling**: Reduces the dimensionality of the feature maps while retaining the most salient information (peak vibrations).

2.  **Long Short-Term Memory (LSTM) Stage**:
    * **Temporal Modeling**: The output from the CNN stage is treated as a sequence and passed through an LSTM layer.
    * **Long-range Dependencies**: The LSTM is specifically designed to capture periodic patterns and long-term temporal dependencies within the 1024-sample window that standard CNNs might overlook.

3.  **Multi-Layer Perceptron (MLP) / Dense Stage**:
    * **Classification Head**: The final hidden state of the LSTM is fed into a fully connected MLP.
    * **Non-linear Transformation**: Uses ReLU activation functions to map the learned features into a high-dimensional space.
    * **Softmax Output**: A final 4-way linear layer with Softmax provides the probability distribution for the four fault classes (Normal, IR, OR, and Ball).

## Usage
1. **Setup**: The script automatically downloads the dataset using kagglehub. Ensure you have an internet connection on the first run.
2. **Training**: Run the notebook cells to train the model. The data is split into 70% Training, 15% Validation, and 15% Testing.

## Visualization:
1. Training/Validation Loss and Accuracy curves are generated.
2. A Confusion Matrix is plotted to evaluate per-class performance.
3. The Grad-CAM cell generates heatmaps overlaid on 1D vibration signals to provide transparency into model decision-making.

## Results
The model achieves high accuracy across all fault types. The following artifacts are generated in the plots/ directory:
- loss_accuracy_curves.png: Training progress.
- confusion_matrix.png: Detailed performance breakdown.
- gradcam_1d_analysis.png: Visual evidence for fault classification.
