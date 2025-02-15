import os
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import sys
import cv2
import numpy as np
import onnxruntime as ort
from types import SimpleNamespace
from collections import deque, namedtuple
import torchvision.transforms as transforms
import torchvision
from PIL import Image
# from insightface.app import FaceAnalysis  # Removed insightface
from facenet_pytorch import MTCNN, InceptionResnetV1  # Added FaceNet PyTorch

# Define a namedtuple to hold match information
Match = namedtuple("Match", ["frame_idx", "track_id", "similarity", "bbox", "frame_image", "feature_source"])

# Change directory back to the main folder
os.chdir("/Volumes/Data Center/UPB/Courses/A2S1/A2S1 - Research 3")

# Step 2: Load YOLOv8 Model for Detection
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = YOLO("yolov8n.pt")
model.to(device)

# Path to the folder containing reference images
reference_image_folder = "/Volumes/Data Center/UPB/Courses/A2S1/A2S1 - Research 3/Ref_img"

# Step 3: Import ByteTrack
sys.path.append('/Volumes/Data Center/UPB/Courses/A2S1/A2S1 - Research 3/ByteTrack')
from yolox.tracker.byte_tracker import BYTETracker

args_dict = {
    "track_thresh": 0.5,
    "high_thresh": 0.6,
    "match_thresh": 0.8,
    "track_buffer": 30,
    "mot20": False
}
args = SimpleNamespace(**args_dict)
tracker = BYTETracker(args)

# Step 4: Load and Configure FastReID
reid_session = ort.InferenceSession("./fast-reid/models/sbs_R101.onnx")
os.chdir("/Volumes/Data Center/UPB/Courses/A2S1/A2S1 - Research 3")

# Initialize FaceNet models (MTCNN for face detection, InceptionResnetV1 for feature extraction)
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20,
                  thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=device)  # Keep device consistent
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Feature Extraction Function (as before)
def extract_features(image):
    img = cv2.resize(image, (128, 384)).astype("float32")
    img = img.transpose(2, 0, 1)[None, :, :, :]
    input_name = reid_session.get_inputs()[0].name
    features = reid_session.run(None, {input_name: img})[0]
    features = features.squeeze() if features.ndim > 1 else features
    features /= np.linalg.norm(features)
    return features

# Feature Extraction Function (FaceNet)
def extract_face_features(image):
   try:
        # Convert the image to PIL format for MTCNN
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Detect faces and landmarks
        boxes, _ = mtcnn.detect(image_pil)

        if boxes is not None and len(boxes) > 0:
            # Extract the first detected face
            face_img = image_pil.crop(boxes[0]) #Cropping is done with PIL, so PIL image format is required

            # Transform the face image and extract features
            face_img_tensor = transforms.ToTensor()(face_img).unsqueeze(0).to(device)  # Convert to tensor and move to device
            with torch.no_grad():
                face_embedding = resnet(face_img_tensor).cpu().numpy().flatten()
            return face_embedding
        else:
            return None
   except Exception as e:
        print(f"Error extracting FaceNet features: {e}")
        return None

# Load reference images from the folder
reference_features = []
reference_image_paths = [os.path.join(reference_image_folder, f) for f in os.listdir(reference_image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith("._")]
with tqdm(total=len(reference_image_paths), desc="Processing Reference Images", bar_format="{l_bar}{bar} [Elapsed: {elapsed} | Remaining: {remaining}]") as pbar:
    for filename in reference_image_paths:
        try:
           reference_img = cv2.imread(filename)
           if reference_img is None:
                raise FileNotFoundError(f"Could not load image: {filename}")
           face_embedding = extract_face_features(reference_img)

           reference_features.append((face_embedding, filename)) # store face features and filename for debugging
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
            reference_features.append((None, filename))

        pbar.update(1)

if not reference_features:
    raise FileNotFoundError(f"No valid image files found in {reference_image_folder}")

# Cosine Similarity Function (as before)
def cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Function to combine features, add a feature attention function here in later revisions
def combine_features(reid_features, difference_map=None, color_hist=None):
     features = []
     if reid_features is not None:
        features.append(reid_features)
     if difference_map is not None:
         features.append(difference_map)
     if color_hist is not None:
        features.append(color_hist)
     return np.concatenate(features) if features else np.array([]) # Combine all the features using concatenation

# Step 6: Real-Time Video Processing
records_folder = "/Volumes/Data Center/UPB/Courses/A2S1/A2S1 - Research 3/Records"
video_files = [f for f in os.listdir(records_folder) if f.lower().endswith(('.mp4', '.MP4', '.mov', '.MOV'))]

# Create the output folder if it doesn't exist
output_folder = "match_photos"
os.makedirs(output_folder, exist_ok=True)

total_videos = len(video_files)
for video_number, video_file in enumerate(video_files):
    video_source = os.path.join(records_folder, video_file)
    print(f"\nProcessing Video {video_number + 1}/{total_videos}: {video_file}")

    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
       print(f"Error: Could not open video {video_file}")
       continue

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) #Get FPS for time calculation
    matches_found_total = 0
    start_time = time.time()
    top_matches = []


    with tqdm(total=frame_count, desc="Processing Frames", bar_format="{l_bar}{bar} [Elapsed: {elapsed} | Remaining: {remaining}]") as pbar:
        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            detections = results[0].boxes.data
            frame_height, frame_width = frame.shape[:2]
            img_info = (frame_height, frame_width)

            try:
                if detections.shape[0] > 0:
                  tracks = tracker.update(torch.tensor([arr.cpu().numpy() for arr in detections]), img_info, (frame_width, frame_height))
                  for track in tracks:
                      if not hasattr(track, 'tlbr'):
                          continue
                      track_id = int(track.track_id)
                      bbox = track.tlbr

                      face_img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                      if face_img.size > 0 and face_img.shape[0] > 0 and face_img.shape[1] > 0:
                        face_embedding = extract_face_features(face_img)
                        feature_source = "FaceNet"
                      else:
                        face_embedding = None
                        feature_source = "None"

                      best_similarity = 0
                      best_match = None
                      best_ref_name = None

                      for ref_face_embedding, ref_name in reference_features:
                          if ref_face_embedding is None or face_embedding is None:
                              continue
                          similarity = cosine_similarity(ref_face_embedding, face_embedding)
                          if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = (bbox, similarity, track_id, frame, ref_name, feature_source)

                      if best_match is not None and best_similarity > 0.3:
                          bbox, similarity, track_id, best_frame, ref_name, feature_source = best_match

                          top_matches.append(Match(frame_idx, track_id, similarity, bbox, best_frame, feature_source))
                          matches_found_total += 1
                          print(f"Similarity Score for Track ID {track_id}: best score {best_similarity:.2f} using image {ref_name} using {feature_source}")

                          # Draw the bounding box on the saved image
                          match_frame_with_box = best_frame.copy()
                          cv2.rectangle(match_frame_with_box, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

                          # Calculate frame time
                          frame_time = frame_idx / fps if fps > 0 else 0  # Avoid dividing by zero
                          time_str = f"{int(frame_time // 60):02}:{int(frame_time % 60):02}" #Format as MM:SS

                          # Save the match photo
                          video_name_part = os.path.splitext(video_file)[0] # Strip Extension
                          match_filename = f"{video_name_part}_time_{time_str}_frame_{frame_idx}_track_{track_id}_feature_{feature_source}.jpg"
                          match_path = os.path.join(output_folder, match_filename)
                          cv2.imwrite(match_path, match_frame_with_box)

            except AttributeError as e:
                print(f"AttributeError: {e}")

            pbar.update(1)

    cap.release()
    end_time = time.time()
    processing_time = end_time - start_time

    print(f"Total Matches Found: {matches_found_total}")
    print(f"Total Frames Processed: {frame_count}")
    print(f"Total Processing Time: {processing_time:.2f} seconds")
    print(f"Average FPS: {frame_count / processing_time:.2f}")

    # Print Top Matches
    print("\nTop Matches:")
    sorted_matches = sorted(top_matches, key=lambda match: match.similarity, reverse=True)

    if sorted_matches:  # Check if there are any matches
        best_match = sorted_matches[0]
        print(f"  Frame: {best_match.frame_idx}, Track ID: {best_match.track_id}, Similarity: {best_match.similarity:.2f}, BBox: {best_match.bbox}, Feature Source: {best_match.feature_source}")
    else:
      print("No matches found")


print(f"\nMatch photos saved to '{output_folder}' folder.")

cv2.destroyAllWindows() # Close all windows
print("\nAll videos processed.")

