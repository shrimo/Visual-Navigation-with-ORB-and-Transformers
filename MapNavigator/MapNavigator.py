import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform

class MapNavigator:
    def __init__(self, map_image_path):
        """
        Initialize the map navigator with an image.
        :param map_image_path: Path to the map image file.
        """
        self.map_image = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)
        if self.map_image is None:
            print(f"Error: Unable to load map image from path: {map_image_path}")
            return

        self.map_image = self.extract_roads(self.map_image)
        self.akaze = cv2.AKAZE_create()
        self.map_keypoints, self.map_descriptors = self.akaze.detectAndCompute(self.map_image, None)
        self.trajectory = []

    @staticmethod
    def extract_roads(image):
        """
        Extract roads from the map image using advanced techniques.
        :param image: Input image.
        :return: Image with only roads highlighted.
        """
        # Apply Gaussian blur to the image
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Use Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Apply morphological operations to close gaps in edges
        kernel = np.ones((5, 5), np.uint8)
        morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return morph

    def match_features(self, frame):
        """
        Match features between the map and the current frame.
        :param frame: Frame from the video.
        :return: Homography matrix if matches are found.
        """
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = self.extract_roads(frame_gray)
        frame_keypoints, frame_descriptors = self.akaze.detectAndCompute(frame_gray, None)
        if frame_descriptors is None or self.map_descriptors is None:
            return None, [], []

        # Use BFMatcher with default params
        matches = bf.match(self.map_descriptors, frame_descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > 10:
            src_pts = np.float32([self.map_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 2)
            dst_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 2)

            kmeans = KMeans(n_clusters=2)
            kmeans.fit(dst_pts)
            clustered_points = dst_pts[kmeans.labels_ == 0]
            clustered_map_points = src_pts[kmeans.labels_ == 0]

            if len(clustered_points) > 4:
                model, inliers = ransac(
                    (clustered_points, clustered_map_points),
                    ProjectiveTransform,
                    min_samples=4,
                    residual_threshold=2,
                    max_trials=1000
                )
                if model is not None:
                    return model.params, clustered_points, clustered_map_points
        return None, [], []

    def smooth_trajectory(self, trajectory, window_size=5):
        """
        Smooth the trajectory using a moving average filter.
        :param trajectory: List of (x, y) positions.
        :param window_size: Size of the moving window.
        :return: Smoothed trajectory.
        """
        if len(trajectory) < window_size:
            return trajectory
        
        smoothed_trajectory = []
        for i in range(len(trajectory)):
            start = max(0, i - window_size // 2)
            end = min(len(trajectory), i + window_size // 2 + 1)
            avg_x = np.mean([pt[0] for pt in trajectory[start:end]])
            avg_y = np.mean([pt[1] for pt in trajectory[start:end]])
            smoothed_trajectory.append((int(avg_x), int(avg_y)))
        return smoothed_trajectory

    def process_frame(self, frame):
        """
        Process a single frame to match features and estimate position.
        :param frame: Frame from the video.
        :return: Processed frame with keypoints and trajectory.
        """
        matrix, matched_pts, map_pts = self.match_features(frame)

        vis_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_GRAY2BGR)

        for pt in matched_pts:
            cv2.drawMarker(vis_frame, tuple(int(v) for v in pt), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, 
                           markerSize=10, thickness=2, line_type=cv2.LINE_AA)
        
        if matrix is not None:
            h, w = frame.shape[:2]
            x, y = np.mean(cv2.perspectiveTransform(np.float32([[w/2, h/2]]).reshape(-1, 1, 2), matrix), axis=0).ravel()
            self.trajectory.append((int(x), int(y)))
            print(f"Estimated drone position on map: ({int(x)}, {int(y)})")

            # Resize the map image for visualization
            resized_map_image = cv2.resize(self.map_image, (self.map_image.shape[1] // 2, self.map_image.shape[0] // 2))

            img = cv2.cvtColor(resized_map_image, cv2.COLOR_GRAY2BGR)

            for pt in map_pts:
                cv2.drawMarker(img, (int(pt[0] // 2), int(pt[1] // 2)), color=(0, 255, 255), markerType=cv2.MARKER_CROSS, 
                               markerSize=10, thickness=2, line_type=cv2.LINE_AA)
            cv2.drawMarker(img, (int(x // 2), int(y // 2)), color=(255, 0, 0), markerType=cv2.MARKER_TILTED_CROSS, 
                           markerSize=15, thickness=3, line_type=cv2.LINE_AA)

            # Smooth the trajectory
            smoothed_trajectory = self.smooth_trajectory(self.trajectory)

            # Draw trajectory
            for i in range(1, len(smoothed_trajectory)):
                pt1 = (smoothed_trajectory[i-1][0] // 2, smoothed_trajectory[i-1][1] // 2)
                pt2 = (smoothed_trajectory[i][0] // 2, smoothed_trajectory[i][1] // 2)
                cv2.line(img, pt1, pt2, (0, 0, 255), 2)

            cv2.imshow("Drone Navigation", img)
        else:
            print("No reliable matches found.")

        return vis_frame

    def match_drone_position(self, video_path):
        """
        Match the drone's position from video feed with the map location.
        :param video_path: Path to the drone video.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video file: {video_path}")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            vis_frame = self.process_frame(frame)
            cv2.imshow("Drone Video", vis_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    navigator = MapNavigator("map.png")
    navigator.match_drone_position("drone_video.mp4")