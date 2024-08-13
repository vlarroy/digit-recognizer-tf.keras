# **Digit recognizer with tf.keras**

## Overview

This project demonstrates the use of machine learning techniques to recognize handwritten digits. The model is trained and evaluated on a dataset of digit images, with each image represented as a 28x28 pixel grid. The notebook walks through the steps of data preprocessing, model training, and prediction.

## Dataset

The dataset includes:

    - train.csv: A file containing thousands of gray-scale images of hand-drawn digits, labeled from zero to nine. Each row in this file represents an image and its corresponding digit label.

    - test.csv: A file with similar gray-scale images, but without labels. This data will be used to evaluate the performance of our trained model.

The Digit Recognizer dataset is available through the Kaggle competition and can be accessed here: [Digit Recognizer Dataset](https://www.kaggle.com/competitions/digit-recognizer/data)


## Files

    - main.ipynb: The main Jupyter notebook that contains all the steps for building, training, and testing the digit recognition model.

## Prerequisites

To run this notebook, you'll need to have the following packages installed:

### Basic Libraries

    NumPy: For numerical computations.
    Pandas: For data manipulation and analysis.

### Visualization Libraries

    Matplotlib: For creating static plots and visualizations.
    Plotly: For interactive visualizations.
    Graphviz: For visualizing decision trees.

### TensorFlow and Keras Libraries

    TensorFlow: For building and training the neural network models.
    Keras: A high-level API for TensorFlow that simplifies building neural networks.
    Keras CV (Computer Vision): For computer vision tasks.

### Scikit-Learn Libraries

    Scikit-Learn: For various machine learning tasks including model training, evaluation, and hyperparameter tuning.

## **Contents of the Notebook**
### 1. Data Loading

The dataset, previously mentioned.

### 2. Data Preprocessing

The notebook processes the pixel values to prepare them for model training. This includes reshaping the data into a 28x28 grid format suitable for image processing.

### 3. Model Building

A TensorFlow-based neural network model is constructed. The architecture likely includes multiple layers, such as Conv2D, MaxPooling2D, and Dense layers, to recognize patterns in the digit images.

### 4. Model Training

The model is trained on the training dataset, and its performance is evaluated using the test dataset. The notebook includes visualizations to monitor the training process.

### 5. Prediction

The trained model is used to predict digits from the test dataset. The notebook demonstrates how to make predictions for individual images as well as the entire test set.

### 6. Visualization

The notebook includes code to visualize the images from the dataset and the corresponding predictions made by the model.