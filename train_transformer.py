import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset
import joblib
from transformer_model import TransformerNavigation
import os

def extract_orb_features(video_path, max_features=500, cache_file="descriptors.npy", visualize=False):
    """
    Extracts ORB features from a video file and caches the results.
    Optionally visualizes the detected features on each frame.
    Args:
        video_path (str): Path to the video file.
        max_features (int): Maximum number of features per frame.
        cache_file (str): Path to the cache file for saving/loading descriptors.
        visualize (bool): If True, display the video frames with keypoints.
    Returns:
        np.ndarray: Extracted ORB descriptors.
    """
    if os.path.isfile(cache_file):
        print(f"Loading cached descriptors from '{cache_file}'...")
        return np.load(cache_file)

    if not os.path.isfile(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return None

    cap = cv2.VideoCapture(video_path)
    orb = cv2.ORB_create(nfeatures=max_features)
    descriptors_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        # Visualize extracted features if requested
        if visualize:
            frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)
            cv2.imshow("ORB Features", frame_with_keypoints)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                visualize = False  # Stop visualization if 'q' is pressed

        if descriptors is not None:
            descriptors_list.append(descriptors)

    cap.release()
    if visualize:
        cv2.destroyAllWindows()

    if not descriptors_list:
        print("Error: No ORB features found in the video.")
        return None

    descriptors = np.vstack(descriptors_list)
    np.save(cache_file, descriptors)
    print(f"Descriptors cached to '{cache_file}'")
    return descriptors

def tokenize_descriptors(descriptors, num_clusters=100):
    """
    Clusters ORB descriptors using K-Means to obtain tokenized features.
    Args:
        descriptors (np.ndarray): ORB descriptors extracted from the video.
        num_clusters (int): Number of clusters for K-Means.
    Returns:
        tuple: Cluster labels and the trained KMeans model.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(descriptors)
    return labels, kmeans

def train_transformer(descriptors, labels, kmeans, epochs=10, batch_size=32, lr=0.0005,
                      model_save_path="transformer_model.pth", kmeans_save_path="kmeans_model.pkl"):
    """
    Trains a Transformer model using tokenized descriptors.
    Instead of using raw descriptors (dimension 32), token indices are used, which are then
    converted to a one-hot representation to match the model's input.
    Added weight decay and a scheduler to reduce the risk of the training loss dropping to zero too quickly.
    
    Args:
        descriptors (np.ndarray): ORB descriptors.
        labels (np.ndarray): Cluster labels.
        kmeans (KMeans): The trained KMeans model.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        lr (float): Learning rate.
        model_save_path (str): Path to save the trained model.
        kmeans_save_path (str): Path to save the KMeans model.
    Returns:
        TransformerNavigation: The trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use token indices with an added sequence dimension (seq_len=1)
    X = torch.tensor(labels, dtype=torch.long).unsqueeze(1).to(device)  # Shape: (num_samples, 1)
    y = torch.tensor(labels, dtype=torch.long).to(device)
    assert X.shape[0] == y.shape[0], "Size mismatch between tokens and labels."

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TransformerNavigation(
        input_dim=kmeans.n_clusters,  # The one-hot representation dimension equals number of clusters
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        num_classes=len(set(labels)),
        dropout=0.3  # Dropout for added regularization
    ).to(device)

    # Adding weight_decay for regularization to reduce the risk of overfitting
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            # Convert token indices to one-hot representation.
            # batch_X has shape (batch, 1) -> squeeze to (batch,) -> one-hot -> (batch, input_dim) -> unsqueeze to (batch, 1, input_dim)
            batch_X_onehot = torch.nn.functional.one_hot(batch_X.squeeze(1), num_classes=kmeans.n_clusters).float().unsqueeze(1)
            outputs = model(batch_X_onehot)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        if avg_loss == 0:
            print("Warning: Training loss reached zero. Check data and model configuration for potential issues.")

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    joblib.dump(kmeans, kmeans_save_path)
    os.chmod(kmeans_save_path, 0o600)  # Restrict access to the KMeans model file
    print(f"K-Means model saved to {kmeans_save_path} with restricted permissions")

    return model

if __name__ == "__main__":
    video_path = "drone_flight.mp4"
    # Set visualize=True to display ORB features during extraction
    print("Extracting ORB features...")
    descriptors = extract_orb_features(video_path, visualize=True)

    if descriptors is None:
        print("Failed to extract ORB features. Exiting...")
        exit(1)

    print("Tokenizing descriptors...")
    labels, kmeans = tokenize_descriptors(descriptors)

    print("Training Transformer model...")
    trained_model = train_transformer(descriptors, labels, kmeans)

    print("Model training completed.")
    print("""Usage Instructions:
    1. Place a video file of a drone flight in the same directory and set its name in 'video_path'.
    2. Run the script. It will extract ORB features and train a transformer model.
    3. The trained model will be saved as 'transformer_model.pth'.
    4. The KMeans model will be saved as 'kmeans_model.pkl' with restricted permissions.
    5. Use the separate inference script to apply the trained model for navigation tasks.""")