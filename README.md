## Sign Language Recognition with CNN
This project uses a Convolutional Neural Network (CNN) to recognize American Sign Language (ASL) gestures.


# Introduction
Sign language recognition is an application of computer vision and machine learning that translates sign language gestures into text or speech. This project focuses on recognizing ASL gestures using a CNN model trained on a custom dataset of ASL hand signs. In this project, i have made datasets for the alphabets A, B, C and L.

# Project Overview
The project is divided into the following main components:

Dataset Collection: Capturing ASL gestures using a webcam and saving them as images.
Model Training: Training a CNN model on the collected dataset to recognize ASL gestures.
Real-time Prediction: Using the trained model to predict ASL gestures in real-time.

# Dataset Collection
To collect the dataset, we use OpenCV to capture hand gestures from a webcam and save them as images. Each gesture is associated with a key on the keyboard, and images are saved in directories corresponding to each gesture.

# Model Training
The collected dataset is used to train a CNN model to recognize ASL gestures. The model is trained on the images saved during the data collection phase.

# Real-time Prediction
The trained CNN model is used to predict ASL gestures in real-time. The webcam captures the hand gestures, and the model predicts the corresponding ASL gesture.

# Contributing
Contributions are welcome! Please feel free to fork the repository and submit pull requests.

# License
This project is licensed under the MIT License - see the LICENSE file for details.
