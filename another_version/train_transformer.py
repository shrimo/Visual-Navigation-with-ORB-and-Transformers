import cv2
import torch
import torch.nn as nn
import numpy as np
import logging
import os
import pickle
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
from transformer_model import TransformerModel
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoDataset(Dataset):
    def __init__(self, video_path, n_clusters=1):
        self.frames = []
        self.descriptors = []
        # Load video frames and extract descriptors using ORB
        cap = cv2.VideoCapture(video_path)
        logging.info("Loading video frames...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            des = self.extract_descriptors(gray)
            if des.shape[0] > 0:
                self.frames.append(des)
                self.descriptors.extend(des)
        cap.release()
        logging.info(f"Total frames loaded: {len(self.frames)}")
        
        # Ensure that there are some descriptors before clustering
        if len(self.descriptors) == 0:
            raise ValueError("No descriptors found. Check the video file path or file content.")

        # Cluster all descriptors using KMeans
        logging.info("Clustering descriptors with KMeans...")
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(np.array(self.descriptors))
        logging.info("Clustering completed.")

    def extract_descriptors(self, gray_frame):
        orb = cv2.ORB_create()
        _, descriptors = orb.detectAndCompute(gray_frame, None)
        if descriptors is None:
            return np.empty((0, 32), dtype=int)
        return descriptors

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        desc = self.frames[idx]
        # Obtain cluster labels for each descriptor using the pretrained KMeans
        labels = self.kmeans.predict(desc)
        # Return as an array with shape (num_descriptors, 1)
        return labels.reshape(-1, 1)

def collate_fn(batch):
    batch_sizes = [item.shape[0] for item in batch]
    max_len = max(batch_sizes)
    # Pad each item to the maximum length
    padded = [np.pad(item, ((0, max_len - item.shape[0]), (0, 0)), mode='constant') for item in batch]
    # Convert list of arrays to a single numpy array before converting to tensor
    padded_array = np.array(padded)
    return torch.from_numpy(padded_array).float()

def generate_dummy_targets(batch_size, seq_len, output_dim):
    return torch.zeros(batch_size, seq_len, output_dim, dtype=torch.float32)

def train_model(model, dataloader, epochs, lr, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        logging.info(f"Starting epoch {epoch+1}/{epochs}")
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)  # output shape: [batch, seq_len, output_dim]
            tgt = generate_dummy_targets(batch.size(0), batch.size(1), model.fc_out.out_features).to(device)
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logging.info(f"Epoch {epoch+1} loss: {total_loss / len(dataloader):.4f}")

if __name__ == "__main__":
    video_path = "../drone_flight.mp4"
    cache_file = "dataset_cache_train.pkl"
    # Check if a cached dataset exists to avoid reprocessing
    if os.path.exists(cache_file):
        logging.info("Loading cached dataset...")
        with open(cache_file, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = VideoDataset(video_path, n_clusters=1)  # input_dim = 1 (cluster label)
        with open(cache_file, "wb") as f:
            pickle.dump(dataset, f)
        logging.info("Dataset cached.")
        
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    input_dim = 1
    model_dim = 64
    num_heads = 8
    num_layers = 6
    output_dim = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)
    epochs = 10
    lr = 0.001
    logging.info("Starting training...")
    train_model(model, dataloader, epochs, lr, device)
    torch.save(model.state_dict(), "transformer_model.pth")
    logging.info("Training completed and model saved.")