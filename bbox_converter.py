import cv2
import os
import glob

input_dir = "/home/sionna/Downloads/1188976"
bboxes_dir = "extracted_bboxes"
output_dir = "extracted_faces_videos"
os.makedirs(output_dir, exist_ok=True)

# Process each bbox file
for bbox_file in glob.glob(os.path.join(bboxes_dir, "*_bboxes.txt")):
    filename = os.path.basename(bbox_file).replace('_bboxes.txt', '')
    # Assuming filename is subdir_video_name
    parts = filename.rsplit('_', 1)
    if len(parts) == 2:
        subdir, video_name = parts
        video_path = os.path.join(input_dir, subdir, video_name + '.mp4')
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            continue
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = os.path.join(output_dir, f"{filename}_faces.mp4")
        out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (224, 224))
        
        with open(bbox_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 6:
                frame_num, x1, y1, x2, y2, conf = parts
                frame_num = int(frame_num)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Seek to the frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
                ret, frame = cap.read()
                if ret:
                    # Crop the face
                    face = frame[y1:y2, x1:x2]
                    if face.size > 0:
                        # Resize to 224x224
                        face_resized = cv2.resize(face, (224, 224))
                        # Write to video
                        out_video.write(face_resized)
        
        cap.release()
        out_video.release()
        print(f"Created video: {output_video_path}")

print("All face videos created.")