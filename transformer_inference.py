import torch
import numpy as np
import cv2
from sklearn.cluster import KMeans
from transformer_model import TransformerNavigation
import joblib

def load_model(model_path, input_dim, hidden_dim=128, num_layers=2, num_heads=4, num_classes=100, dropout=0.3):
    """
    Loads a trained Transformer model from a checkpoint.
    Args:
        model_path (str): Path to the trained model file.
        input_dim (int): Dimension of the one-hot input tokens (number of clusters).
        hidden_dim (int): Dimension of the hidden layers.
        num_layers (int): Number of transformer encoder layers.
        num_heads (int): Number of attention heads.
        num_classes (int): Number of output classes.
        dropout (float): Dropout rate for regularization.
    Returns:
        TransformerNavigation: The loaded Transformer model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerNavigation(input_dim, hidden_dim, num_layers, num_heads, num_classes, dropout=dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def extract_orb_features(frame, max_features=500):
    """
    Extracts ORB features from a video frame.
    Args:
        frame (numpy.ndarray): The video frame image.
        max_features (int): Maximum number of ORB features to detect.
    Returns:
        tuple: A tuple containing keypoints and descriptors.
    """
    orb = cv2.ORB_create(nfeatures=max_features)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, (descriptors if descriptors is not None else np.array([]))

def tokenize_descriptors(descriptors, kmeans_model_path):
    """
    Uses a pre-trained K-Means model to tokenize ORB descriptors.
    Converts cluster labels to a one-hot representation.
    Args:
        descriptors (numpy.ndarray): Extracted ORB descriptors.
        kmeans_model_path (str): Path to the saved KMeans model.
    Returns:
        numpy.ndarray: One-hot representation of the cluster labels.
    """
    if descriptors.size == 0:
        return np.array([])  # Return an empty array if no descriptors are found
    kmeans = joblib.load(kmeans_model_path)
    labels = kmeans.predict(descriptors)
    num_clusters = kmeans.n_clusters
    one_hot_labels = np.eye(num_clusters)[labels]
    return one_hot_labels

def predict_navigation(model, descriptors):
    """
    Predicts a navigation action based on tokenized ORB descriptors.
    Args:
        model (TransformerNavigation): The trained Transformer model.
        descriptors (numpy.ndarray): One-hot tokenized descriptors.
    Returns:
        numpy.ndarray: Predicted class as a numpy array.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Convert the descriptors to a tensor and add a batch dimension.
    # Expected shape: (1, sequence_length, input_dim)
    X = torch.tensor(descriptors, dtype=torch.float32).to(device).unsqueeze(0)
    with torch.no_grad():
        output = model(X)
        predicted_class = torch.argmax(output, dim=1).cpu().numpy()
    return predicted_class

if __name__ == "__main__":
    video_path = "drone_flight.mp4"
    model_path = "transformer_model.pth"
    kmeans_path = "kmeans_model.pkl"

    print("Loading model...")
    kmeans = joblib.load(kmeans_path)
    input_dim = kmeans.n_clusters
    model = load_model(model_path, input_dim=input_dim)

    print("Starting video capture...")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        print("Extracting ORB features from frame...")
        keypoints, descriptors = extract_orb_features(frame)
        if descriptors.size == 0:
            print("No features extracted! Continuing to next frame.")
            continue

        print("Tokenizing descriptors...")
        tokenized_descriptors = tokenize_descriptors(descriptors, kmeans_path)
        if tokenized_descriptors.size == 0:
            print("Tokenization failed! Continuing to next frame.")
            continue

        print("Predicting navigation action...")
        action = predict_navigation(model, tokenized_descriptors)

        frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(255, 0, 0), flags=0)
        cv2.putText(frame_with_keypoints, f"Predicted Navigation Action: {action[0]}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Drone Navigation', frame_with_keypoints)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()