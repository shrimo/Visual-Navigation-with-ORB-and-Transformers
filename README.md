# Visual Navigation Using ORB, Feature Descriptors, and Transformers

This project implements visual navigation using ORB (Oriented FAST and Rotated BRIEF), feature descriptors, and the Transformer deep learning model. Below is a detailed explanation of each step involved in the process.

## ORB (Oriented FAST and Rotated BRIEF)

ORB is a method used for detecting and describing key points in an image. It consists of two main parts:
1. **FAST (Features from Accelerated Segment Test)**: This algorithm is used to detect key points in the image.
2. **BRIEF (Binary Robust Independent Elementary Features)**: This algorithm describes these key points with feature vectors, considering their orientation (Rotated BRIEF).

## Feature Descriptors

After detecting the key points using ORB, each of them is described by a feature vector. These vectors contain information about the local patterns around the key point, making them useful for matching between images.

## Tokenization for the Transformer

The feature descriptors obtained from ORB are processed and tokenized for use in the Transformer model. Visual data is transformed into a sequence of tokens that can be processed by the Transformer model. Each token represents a numerical representation of the local features of the image.

## Transformer Model Training

The Transformer is a powerful deep learning architecture initially designed for processing sequential data such as text. In the case of visual navigation, the tokens obtained from image features are fed into the Transformer model. The model is trained on sequences of tokens to learn how to recognize and match image features, allowing it to build a map of the surrounding space.

## Applying the Model for Navigation

Once trained, the Transformer model can be used for navigation. The robot's camera captures images, which are processed by ORB to obtain feature vectors. These vectors are transformed into tokens that are fed into the Transformer model. The model determines the robot's position and orientation relative to the map, enabling it to navigate the space effectively.

By combining ORB for feature extraction, tokenization for data preparation, and the Transformer model for training and predictions, this system creates a powerful solution for visual navigation.




# Visual Navigation using ORB and Transformer

## Overview
This project implements a visual navigation system using ORB (Oriented FAST and Rotated BRIEF) feature descriptors and a Transformer-based deep learning model. The system is trained on video footage captured by a drone and can be used for navigation tasks based on extracted visual features.

## Installation
### Prerequisites
Ensure you have the following dependencies installed:
```bash
pip install opencv-python torch torchvision scikit-learn numpy joblib
```

## Usage
### Training the Model
1. Place the video file of a drone flight in the working directory.
2. Update the `video_path` variable in `train_transformer.py` to the path of the video.
3. Run the script:
   ```bash
   python train_transformer.py
   ```
4. The trained model will be saved as `transformer_model.pth`.

### Applying the Model for Navigation
1. Ensure `transformer_model.pth` is available in the working directory.
2. Run the inference script:
   ```bash
   python transformer_inference.py
   ```
3. The script will process an input video and output navigation data.

## File Structure
```
.
├── train_transformer.py  # Extracts ORB features, tokenizes, and trains the Transformer model
├── transformer_inference.py  # Applies the trained model for navigation tasks
├── transformer_model.pth  # Saved trained model
├── kmeans_model.pkl # Saved kmeans model
├── drone_flight.mp4  # Sample input video file
└── README.md  # Documentation
```

## Model Training Process
- **Feature Extraction:** ORB features are extracted from the video frames.
- **Feature Tokenization:** Descriptors are clustered using K-Means.
- **Model Training:** A Transformer-based model is trained on the clustered descriptors.

## Inference Process
- **Load Model:** The trained model is loaded from `transformer_model.pth`.
- **Feature Extraction:** New video frames are processed to extract ORB features.
- **Navigation Prediction:** The model predicts navigation actions based on visual input.

## Notes
- The model requires a sufficient amount of training data to generalize well.
- Ensure the video frames contain distinguishable features for better navigation results.
- The inference script can be modified to integrate real-time drone control.

## License
This project is released under the MIT License.

