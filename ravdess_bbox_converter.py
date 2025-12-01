import cv2
import os
import glob
import multiprocessing

input_dir = "/home/sionna/Downloads/1188976"
bboxes_dir = "/home/sionna/evan/multi-modal-emotion-recognition/extracted_bboxes"
output_dir = "extracted_faces_videos"
os.makedirs(output_dir, exist_ok=True)

def process_video(bbox_file):
    filename = os.path.basename(bbox_file).replace('_bboxes.txt', '')
    # Assuming filename is subdir_video_name
    parts = filename.rsplit('_', 1)
    if len(parts) == 2:
        subdir, video_name = parts
        video_path = os.path.join(input_dir, subdir, video_name + '.mp4')
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            return
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = os.path.join(output_dir, f"{filename}_faces.mp4")
        out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (224, 224))
        
        with open(bbox_file, 'r') as f:
            lines = f.readlines()
        
        # Collect bboxes by frame number
        bboxes_by_frame = {}
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 6:
                frame_num, x1, y1, x2, y2, conf = parts
                frame_num = int(frame_num)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                bboxes_by_frame[frame_num] = (x1, y1, x2, y2)
        
        # Read video frame by frame
        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
            
            if frame_num in bboxes_by_frame:
                x1, y1, x2, y2 = bboxes_by_frame[frame_num]
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

# Get list of bbox files
bbox_files = glob.glob(os.path.join(bboxes_dir, "*_bboxes.txt"))

# Use multiprocessing to process videos in parallel
num_processes = min(multiprocessing.cpu_count(), len(bbox_files))
with multiprocessing.Pool(processes=num_processes) as pool:
    pool.map(process_video, bbox_files)

print("All face videos created.")