Flower Image Classifier Project
GitHub Repository: https://github.com/ahmadyacoub/Flower_Image_Classifier_Project• Developed an image classification system to identify 102 flower species using the Oxford Flowers 102 dataset.• Built with TensorFlow and TensorFlow Hub, leveraging transfer learning with a pre-trained MobileNet model.• Implemented a Jupyter Notebook for model training and evaluation, and a command-line application (predict.py) for inference with top-K predictions and flower name mapping.• Completed as part of the AI Programming with Python and TensorFlow course through the Palestine Launchpad Initiative by Google and Udacity.

Project Overview
This project builds an image classifier to identify 102 flower species using the Oxford Flowers 102 dataset. It leverages TensorFlow and transfer learning with a pre-trained MobileNet model from TensorFlow Hub. The project includes a Jupyter Notebook for model development and a command-line application (predict.py) for inference.
Components

Development Notebook: Project_Image_Classifier_Project.ipynb for data loading, model training, evaluation, and visualization.
Command-Line Application: predict.py for classifying new images using the trained model.

Rubric Compliance
The project fulfills all requirements of the "Create Your Own Image Classifier - TensorFlow" rubric:
Part 1 - Development Notebook

Package Imports: Imports TensorFlow, TensorFlow Hub, NumPy, Matplotlib, etc.
Data Loading: Loads Oxford Flowers 102 via TensorFlow Datasets.
Data Splits: Splits dataset into training, validation, and test sets.
Dataset Info: Prints number of examples and 102 classes.
Dataset Images: Displays shapes and labels of first three training images.
Plot Image: Plots first training image with label as title.
Label Mapping: Plots first image with class name from label_map.json.
Data Normalization: Resizes images to 224x224 and normalizes to [0, 1].
Data Pipeline: Constructs pipelines with resizing and normalization.
Data Batching: Returns batched images for training.
Pre-trained Network: Uses MobileNet with frozen weights.
Feedforward Classifier: Adds custom network with 102 output neurons.
Training the Network: Compiles and trains model with validation data.
Validation Loss and Accuracy: Plots training/validation metrics.
Testing Accuracy: Evaluates model on test set.
Saving the Model: Saves model as trained_model.h5.
Loading Model: Loads saved model for inference.
Image Processing: process_image resizes and normalizes images to (224, 224, 3).
Inference: predict function returns top-K classes.
Sanity Check: Visualizes top 5 predicted classes with flower names.

Part 2 - Command Line Application

Predicting Classes: predict.py outputs most likely class and probability.
Top K Classes: Supports top-K classes (default K=5).
Displaying Class Names: Maps class indices to flower names using label_map.json.

Enhancements

Regularization: Uses dropout to keep training-validation accuracy gap <3%.
Sanity Checking: Visualizations confirm accurate predictions.

Prerequisites

Python 3.8+
TensorFlow 2.x
TensorFlow Hub
TensorFlow Datasets
NumPy
Matplotlib
Pillow (PIL)
Jupyter Notebook

Installation

Clone the Repository:
git clone https://github.com/ahmadyacoub/Flower_Image_Classifier_Project.git
cd Flower_Image_Classifier_Project


Create a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install tensorflow tensorflow-hub tensorflow-datasets numpy matplotlib pillow


Install Jupyter Notebook:
pip install jupyter




Note: The trained_model.h5 file (13MB) is not included to keep the repository lightweight. Retrain the model using the notebook or download it from Google Drive link placeholder.
Usage
Part 1: Development Notebook

Start Jupyter Notebook:
jupyter notebook


Open Project_Image_Classifier_Project.ipynb.

Run cells to:

Load and preprocess data.
Train and save the model.
Evaluate performance.
Classify sample images (e.g., test_images/wild_pansy.jpg).


View outputs:

Images with labels/class names.
Training/validation loss and accuracy plots.
Top 5 predicted classes for a test image.



Part 2: Command-Line Application
Classify images using predict.py.
Basic Usage:
python predict.py test_images/wild_pansy.jpg trained_model.h5

With Top-K Classes:
python predict.py test_images/wild_pansy.jpg trained_model.h5 --top_k 5

With Class Names:
python predict.py test_images/wild_pansy.jpg trained_model.h5 --top_k 5 --category_names label_map.json

Example Output:
Probabilities: [0.85 0.10 0.03 0.01 0.01]
Classes: ['21' '45' '78' '12' '90']
Flower Names: ['wild pansy', 'daisy', 'sunflower', 'rose', 'tulip']

Dataset
The Oxford Flowers 102 dataset includes:

102 classes of flowers.
Training set: 1,020 images.
Validation set: 1,020 images.
Test set: 6,149 images.

Images are resized to 224x224 and normalized for MobileNet.
Model Architecture

Base Model: MobileNet (pre-trained on ImageNet), frozen weights.
Custom Head: Feedforward network with:
Dropout for regularization.
Dense layer with 102 neurons (softmax).


Training: Adam optimizer, categorical crossentropy loss, accuracy metric.

Results

Training/Validation: High accuracy, <3% gap via dropout.
Test Accuracy: Strong generalization on test set.
Inference: Accurate predictions, verified by sanity checks.


License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Oxford Flowers 102 dataset by the University of Oxford.
TensorFlow and TensorFlow Hub for model development.
Part of a deep learning and transfer learning exercise.

