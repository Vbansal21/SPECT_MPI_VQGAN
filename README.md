# Myocardial Perfusion Imaging Analysis and Diagnosis

This project implements a **Myocardial Perfusion Imaging Analysis and Diagnosis** system using machine learning and deep learning models. It focuses on classifying myocardial perfusion scans into normal and abnormal categories while leveraging state-of-the-art techniques like Variational Autoencoders (VAE) and GANs.

---

## Table of Contents

1. [Overview](#overview)
2. [Scope of the Project](#scope-of-the-project)
3. [Current Status](#current-status)
4. [Achievements](#achievements)
5. [Challenges and Limitations](#challenges-and-limitations)
6. [Future Improvements](#future-improvements)
7. [Features](#features)
8. [Installation](#installation)
9. [Usage](#usage)
10. [File Descriptions](#file-descriptions)
11. [Team Members](#team-members)
12. [References](#references)

---

## Overview

Myocardial Perfusion Imaging (MPI) is a diagnostic imaging modality widely used to evaluate blood flow to the heart muscle. This project aims to automate the analysis and diagnosis of MPI scans, offering scalable solutions for clinical and research applications.

---

## Scope of the Project

The scope of this project includes:

- Automating the classification of MPI scans into normal and abnormal categories.
- Improving the efficiency and accuracy of myocardial perfusion imaging diagnosis through advanced AI methods.
- Providing a modular and extensible framework that can be adapted for other medical imaging tasks.
- Enabling clinicians to leverage AI for faster and more accurate diagnoses.

---

## Current Status

- **Data Preprocessing**: Completed with the implementation of scripts like `ImgPP.py` for segmentation and normalization.
- **Model Training**: Achieved with custom implementations of ResNet, VQVAE-GAN, and TinyCLIP-based models.
- **Evaluation Metrics**: Successfully integrated accuracy, precision, recall, F1-score, and confusion matrix for robust performance tracking.
- **Deployment Readiness**: Basic pipeline prepared, but further testing and validation are needed for real-world deployment.

---

## Achievements

1. **High Accuracy Models**: Developed custom architectures that achieve robust performance on synthetic and real-world datasets.
2. **Automated Preprocessing**: Implemented efficient preprocessing pipelines for medical imaging data.
3. **Integrated VQVAE-GAN**: Applied novel GAN techniques for better anomaly detection and feature extraction.
4. **Scalable Architecture**: Designed a modular framework that allows integration of additional features.

---

## Challenges and Limitations

1. **Data Availability**: Limited real-world MPI datasets posed challenges for model training and validation.
2. **Training Stability**: GAN-based models required significant fine-tuning to achieve convergence.
3. **Generalizability**: Ensuring the model performs well across diverse imaging modalities and patient demographics remains a challenge.

---

## Future Improvements

1. **Incorporate Larger Datasets**: Collaborate with healthcare institutions to obtain larger and more diverse datasets.
2. **Real-time Analysis**: Optimize the framework for faster inference and deployment in clinical settings.
3. **Explainability**: Integrate model interpretability features to make predictions more transparent for clinicians.
4. **Integration with PACS**: Enable seamless integration with Picture Archiving and Communication Systems used in hospitals.

---

## Features

- **Preprocessing**: Automatic segmentation and transformation of MPI scan data.
- **Model Architectures**:
  - **ResNet** for classification.
  - **VQVAE-GAN** for reconstruction and anomaly detection.
  - **TinyCLIP** for transfer learning.
- **Custom Loss Functions**:
  - VQ Loss, KL Divergence, Reconstruction Loss.
- **Training Pipeline**: Mixed precision, gradient accumulation, and modular configurations.
- **Evaluation**: Detailed performance metrics for test data.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements-training.txt
   ```

---

## Usage

1. **Preprocess Images**:
   Place raw data in the `dataset/` directory and run:
   ```bash
   python ImgPP.py
   ```

2. **Train the Model**:
   Configure parameters in `main_script.py` and run:
   ```bash
   python main_script.py
   ```

3. **Evaluate Performance**:
   Use test datasets for performance evaluation:
   ```bash
   python main_script.py --test
   ```

---

## File Descriptions

- **`ImgPP.py`**:
  - Handles image segmentation and preprocessing.
- **`main_script.py`**:
  - Main training pipeline.
- **`resnet.py`**:
  - Implements a ResNet-based model with custom configurations.
- **`tiny_clip.py`**:
  - Incorporates TinyCLIP for feature extraction.
- **`alt_script.py` & `script2d.py`**:
  - Variations of VQVAE-GAN implementations.
- **`requirements-training.txt`**:
  - Dependencies for the project.

---

## Team Members

- **Vaibhav Bansal**
- Additional Team Members:
  - **Ibhaan Agarwal**
  - **Saransh Suri**

---

## References

- This project was guided by **Dr. Varun Tiwari** as part of the **B.Tech in CSE-AIML** curriculum at Manipal University Jaipur.
- Tools and Libraries: PyTorch, PyTorch Lightning, scikit-learn, HuggingFace Transformers, etc.

---

README generated using ChatGPT.
