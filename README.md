# Flower Image Classifier Project

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Issues](https://img.shields.io/github/issues/ahmadyacoub/Flower_Image_Classifier_Project)](https://github.com/ahmadyacoub/Flower_Image_Classifier_Project/issues)

This project builds an image classifier to identify 102 flower species using the [Oxford Flowers 102 dataset](https://www.tensorflow.org/datasets/catalog/oxford_flowers102). It leverages TensorFlow and transfer learning with a pre-trained MobileNet model from TensorFlow Hub. The project includes a Jupyter Notebook for model development and a command-line application (`predict.py`) for inference.

## Project Overview

The project creates an end-to-end image classification pipeline that:
- Loads and preprocesses the Oxford Flowers 102 dataset.
- Trains a deep learning model using transfer learning.
- Saves the trained model for reuse.
- Provides a command-line tool to predict flower species, including top-K probabilities and flower names via a JSON mapping.

### Components
1. **Development Notebook**: `Project_Image_Classifier_Project.ipynb` for data loading, model training, evaluation, and visualization.
2. **Command-Line Application**: `predict.py` for classifying new images using the trained model.

## Rubric Compliance

The project fulfills all requirements of the "Create Your Own Image Classifier - TensorFlow" rubric:

### Part 1 - Development Notebook
- **Package Imports**: Imports TensorFlow, TensorFlow Hub, NumPy, Matplotlib, etc.
- **Data Loading**: Loads Oxford Flowers 102 via TensorFlow Datasets.
- **Data Splits**: Splits dataset into training, validation, and test sets.
- **Dataset Info**: Prints number of examples and 102 classes.
- **Dataset Images**: Displays shapes and labels of first three training images.
- **Plot Image**: Plots first training image with label as title.
- **Label Mapping**: Plots first image with class name from `label_map.json`.
- **Data Normalization**: Resizes images to 224x224 and normalizes to [0, 1].
- **Data Pipeline**: Constructs pipelines with resizing and normalization.
- **Data Batching**: Returns batched images for training.
- **Pre-trained Network**: Uses MobileNet with frozen weights.
- **Feedforward Classifier**: Adds custom network with 102 output neurons.
- **Training the Network**: Compiles and trains model with validation data.
- **Validation Loss and Accuracy**: Plots training/validation metrics.
- **Testing Accuracy**: Evaluates model on test set.
- **Saving the Model**: Saves model as `trained_model.h5`.
- **Loading Model**: Loads saved model for inference.
- **Image Processing**: `process_image` resizes and normalizes images to (224, 224, 3).
- **Inference**: `predict` function returns top-K classes.
- **Sanity Check**: Visualizes top 5 predicted classes with flower names.

### Part 2 - Command Line Application
- **Predicting Classes**: `predict.py` outputs most likely class and probability.
- **Top K Classes**: Supports top-K classes (default K=5).
- **Displaying Class Names**: Maps class indices to flower names using `label_map.json`.

### Enhancements
- **Regularization**: Uses dropout to keep training-validation accuracy gap <3%.
- **Sanity Checking**: Visualizations confirm accurate predictions.

## Prerequisites

- Python 3.8+
- TensorFlow 2.x
- TensorFlow Hub
- TensorFlow Datasets
- NumPy
- Matplotlib
- Pillow (PIL)
- Jupyter Notebook

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ahmadyacoub/Flower_Image_Classifier_Project.git
   cd Flower_Image_Classifier_Project
