# Install once:
# pip install ultralytics opencv-python

import cv2
import os
from pathlib import Path
import glob
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import torch

# Download and load the pre-trained YOLO face detection model from Hugging Face
os.makedirs("models", exist_ok=True)
model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt", local_dir="models")
model = YOLO(model_path)  # Load the model
model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Move to GPU if available

input_dir = "/home/sionna/Downloads/1188976"  # <-- Change this to the directory containing child folders with videos
output_bboxes_folder = "extracted_bboxes"  # Where to save bounding box data
os.makedirs(output_bboxes_folder, exist_ok=True)

# Process each child folder
for subdir in os.listdir(input_dir):
    subdir_path = os.path.join(input_dir, subdir)
    if os.path.isdir(subdir_path):
        # Find all .mp4 videos in the subfolder (adjust extension if needed)
        video_files = glob.glob(os.path.join(subdir_path, "*.mp4"))
        for video_path in video_files:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Could not open video: {video_path}")
                continue
            
            bboxes = []
            frame_number = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_number += 1
                
                # Detect faces using YOLO
                results = model(frame, conf=0.5)
                
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = box.conf[0].item()
                        
                        # Store bounding box data
                        bboxes.append(f"{frame_number},{int(x1)},{int(y1)},{int(x2)},{int(y2)},{confidence:.2f}")
            
            cap.release()
            
            # Save bounding boxes to a text file
            video_name = os.path.basename(video_path).replace('.mp4', '')
            output_file = os.path.join(output_bboxes_folder, f"{subdir}_{video_name}_bboxes.txt")
            with open(output_file, 'w') as f:
                f.write('\n'.join(bboxes))
            
            print(f"Processed {video_path}, saved bboxes to {output_file}")

print("All videos processed.")