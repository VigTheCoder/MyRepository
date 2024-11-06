## Real Time Covid-19 Face Mask Detection
This project is a face mask detection system that can detect if a person is wearing a face mask or not in real-time video streams or images. It uses a deep learning model trained with Keras and TensorFlow, along with OpenCV for face detection, to determine whether each detected face has a mask or not.

## Table of Contents:
- [Overview](#overview)
- [Features](#features)
- [Setup and Requirements](#setup-and-requirements)
- [Dataset](#dataset)
- [Training the Model](#training-the-model)
- [Running the Detection](#running-the-detection)
- [Results](#results)
- [References](#references)

## Overview
Face Mask Detection is a two-part application:

1. **Model Training**: Trains a custom model to classify faces with and without masks.
2. **Real-time Mask Detection**: Uses the trained model to detect face masks on people in real-time from a webcam feed.

The goal is to create a reliable face mask detector to help ensure public safety in areas requiring mask compliance.

## Features
- **Real-time Detection**: Detects face masks in a live video feed.
- **High Accuracy**: Trained on a custom dataset to ensure accuracy in detection.
- **MobileNetV2 Backbone**: Uses the lightweight MobileNetV2 model for efficient performance.
- **Data Augmentation**: Enhances the model's robustness by augmenting training images with rotations, shifts, and more.

## Setup and Requirements

Prerequisites
To get started, you’ll need to have the following installed:
- Python 3.7+
- TensorFlow
- Keras
- OpenCV
- imutils

To install all dependencies, run `pip install tensorflow keras opencv-python imutils numpy matplotlib` in your terminal.

Project Directory Layout

    .
    ├── build                    # Compiled files (alternatively `dist`)
    ├── dataset                  # Training dataset
    │   ├── with_mask            # Images of people wearing masks
    │   └── without_mask         # Images of people without masks
    ├── docs                     # Documentation files
    ├── face_detector            # Pre-trained face detection model files
    │   ├── deploy.prototxt      # Model architecture file
    │   └── res10_300x300_ssd_iter_140000.caffemodel  # Model weights file
    ├── src                      # Source files for mask detection and training
    │   ├── detect_mask_video.py # Script for real-time mask detection
    │   └── train_mask_detector.py  # Script for training the mask detection model
    ├── test                     # Automated tests for model accuracy and detection
    ├── tools                    # Utilities and additional helper scripts
    ├── mask_detector.model      # Saved trained model for mask detection
    ├── LICENSE
    └── README.md

> Use short lowercase names at least for the top-level files and folders except
> `LICENSE`, `README.md`

 ## Dataset
The dataset should have two categories in separate folders:

with_mask: Images of people wearing masks.
without_mask: Images of people without masks.
The dataset used for training should be placed in the dataset/ directory.

## Training the Model
To train the face mask detection model, execute the command `python train_mask_detector.py`. 

This script will
Load the dataset.
Preprocess and augment the data.
Train the MobileNetV2-based model.
Save the trained model as mask_detector.model.
Parameters
INIT_LR: Initial learning rate.
EPOCHS: Number of training epochs.
BS: Batch size.
Once trained, the model will be saved as mask_detector.model for later use.

## Running the Detection
To start the real-time face mask detection, run the detect_mask_video.py script:
python detect_mask_video.py

This script will
Load the trained model (mask_detector.model).
Initialize the video stream from the webcam.
Detect faces using OpenCV’s faceNet.
Predict if a face is wearing a mask or not using maskNet.
Press q to exit the real-time video stream.

## Results
The script outputs a real-time video feed where each detected face is surrounded by a bounding box and labeled with:
Mask: Green bounding box.
No Mask: Red bounding box.
It also shows the confidence level of each prediction.

Model Performance
After training, you will receive a classification report and accuracy/loss graphs. Adjust EPOCHS, INIT_LR, and other parameters in train_mask_detector.py as needed to improve accuracy.

## References
OpenCV: https://opencv.org/
TensorFlow and Keras: https://www.tensorflow.org/

