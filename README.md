# DreamLens — Transformer-Based EEG Sleep Stage Classification

DreamLens is an AI-based final year project focused on classifying EEG sleep stages using deep learning. The system classifies sleep into Wake, N1, N2, N3, and REM stages based on AASM sleep-stage standards.

## Project Overview

Manual sleep stage scoring can be time-consuming and dependent on expert analysis. DreamLens explores an efficient Transformer-based approach for automatic EEG sleep stage classification using EEG signal data.

## Aim

The aim of this project is to develop a lightweight Transformer-based model that can classify EEG sleep stages while maintaining strong performance and computational efficiency.

## Dataset

This project uses the Sleep-EDF Expanded dataset from PhysioNet.

Dataset files are not included in this repository due to size and usage restrictions. Users should download the dataset from the official PhysioNet source.

## Sleep Stages

The model classifies the following stages:

- Wake
- N1
- N2
- N3
- REM

## Methodology

The project follows these steps:

1. Load EEG recordings
2. Preprocess EEG signals
3. Segment EEG data into epochs
4. Train the Transformer-based model
5. Evaluate model performance
6. Visualize results using a dashboard

## Technologies Used

- Python
- NumPy
- Pandas
- SciPy
- MNE-Python
- PyTorch / TensorFlow
- scikit-learn
- Streamlit
- Matplotlib / Plotly

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Macro F1-Score
- Cohen’s Kappa
- Confusion Matrix

## Dashboard

The dashboard is designed to display:

- Predicted sleep stages
- Hypnogram visualization
- Sleep-stage distribution
- Model evaluation results

## Screenshots

Add screenshots here:

![Dashboard Screenshot](assets/dashboard_screenshot.png)

![Sample Hypnogram](assets/sample_hypnogram.png)

## How to Run

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/dreamlens-eeg-sleep-stage-classification.git
cd dreamlens-eeg-sleep-stage-classification
