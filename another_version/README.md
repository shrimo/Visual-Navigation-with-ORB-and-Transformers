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



# Transformer-based Navigation System

This project implements a Transformer-based navigation system using PyTorch and OpenCV. The system processes video footage captured by a drone and predicts navigational coordinates.

## Installation

To install the necessary dependencies, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/transformer-navigation.git
    cd transformer-navigation
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Transformer Model

To train the Transformer model using a video file captured by a drone, run the `train_transformer.py` script:

```bash
python train_transformer.py
```

This will process the video, extract ORB features, and train the Transformer model on the extracted features. The trained model will be saved to `transformer_model.pth`.

### Running Inference

To run inference on a new video using the trained Transformer model, use the `transformer_inference.py` script:

```bash
python transformer_inference.py
```

This will process the video, extract ORB features, and use the trained Transformer model to predict navigational coordinates. The results will be displayed on the video with keypoints and predicted positions.

## Project Structure

- `transformer_model.py`: Defines the Transformer model architecture.
- `train_transformer.py`: Script for training the Transformer model using video footage.
- `transformer_inference.py`: Script for running inference using the trained Transformer model.
- `requirements.txt`: Lists the required Python packages.

## Requirements

- Python 3.8 or higher
- PyTorch
- OpenCV
- tqdm

Install these requirements using the following command:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project was developed using PyTorch and OpenCV. We thank the developers and the community for their contributions to these open-source projects.