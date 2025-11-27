# Install once:
# pip install ultralytics opencv-python

import cv2
import os
from pathlib import Path
import glob

# Load the pre-trained face detection model using OpenCV Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                for (x, y, w, h) in faces:
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    confidence = 1.0  # Haar cascades don't provide confidence, set to 1.0
                    
                    # Store bounding box data
                    bboxes.append(f"{frame_number},{x1},{y1},{x2},{y2},{confidence:.2f}")
            
            cap.release()
            
            # Save bounding boxes to a text file
            video_name = os.path.basename(video_path).replace('.mp4', '')
            output_file = os.path.join(output_bboxes_folder, f"{subdir}_{video_name}_bboxes.txt")
            with open(output_file, 'w') as f:
                f.write('\n'.join(bboxes))
            
            print(f"Processed {video_path}, saved bboxes to {output_file}")

print("All videos processed.")