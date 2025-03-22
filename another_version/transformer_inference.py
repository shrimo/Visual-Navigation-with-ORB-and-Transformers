import cv2
import torch
import numpy as np
import logging
import os
import pickle
from sklearn.cluster import KMeans
from transformer_model import load_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Simplified dataset class used for clustering descriptors during inference
class VideoDataset:
    def __init__(self, video_path, n_clusters=1):
        self.descriptors = []
        cap = cv2.VideoCapture(video_path)
        logging.info("Loading video frames for clustering...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            des = self.extract_descriptors(gray)
            if des.shape[0] > 0:
                self.descriptors.extend(des)
        cap.release()
        logging.info(f"Total descriptors collected: {len(self.descriptors)}")
        logging.info("Clustering descriptors with KMeans...")
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(np.array(self.descriptors))
        logging.info("Clustering completed.")

    def extract_descriptors(self, gray_frame):
        orb = cv2.ORB_create()
        _, descriptors = orb.detectAndCompute(gray_frame, None)
        if descriptors is None:
            return np.empty((0, 32), dtype=int)
        return descriptors

def orb_features(frame):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(frame, None)
    if descriptors is None or len(descriptors) == 0:
        descriptors = np.zeros((1, 32), dtype=np.uint8)
    return keypoints, descriptors

def infer_video(model, video_path, device, kmeans):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb_features(gray)
        if descriptors is None or len(descriptors) == 0:
            continue
        # Get cluster labels and convert to tensor with shape [1, seq_len, 1]
        cluster_labels = kmeans.predict(descriptors)
        inp = torch.tensor(cluster_labels.reshape(-1, 1), dtype=torch.float32).unsqueeze(0).to(device)
        output = model(inp)
        pos_text = f"Position: x={output[0][0,0].item():.2f}, y={output[0][0,1].item():.2f}"
        cv2.putText(frame, pos_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        cv2.imshow("Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "drone_footage.mp4"
    model_path = "transformer_model.pth"
    cache_file = "dataset_cache_inference.pkl"
    input_dim = 1
    model_dim = 64
    num_heads = 8
    num_layers = 6
    output_dim = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, input_dim, model_dim, num_heads, num_layers, output_dim, device)
    
    # Check if a cached dataset for inference exists
    if os.path.exists(cache_file):
        logging.info("Loading cached inference dataset...")
        with open(cache_file, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = VideoDataset(video_path, n_clusters=1)
        with open(cache_file, "wb") as f:
            pickle.dump(dataset, f)
        logging.info("Inference dataset cached.")
    
    kmeans = dataset.kmeans
    logging.info(f"Starting inference on video: {video_path}")
    infer_video(model, video_path, device, kmeans)