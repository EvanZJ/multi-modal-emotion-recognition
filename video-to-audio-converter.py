# video_to_audio_batch.py
import os
import ffmpeg  # Correct import for ffmpeg-python package

def video_to_audio(video_path, output_path=None, bitrate="320k"):
    """
    Extract audio from any video file using ffmpeg-python (super fast & reliable)
    """
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return False

    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + ".mp3"

    print(f"Extracting → {output_path}")

    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_path, acodec="libmp3lame", audio_bitrate=bitrate, vn=None)  # vn = no video (fixed to None for flag)
            .overwrite_output()  # overwrite without asking
            .run(quiet=False)  # Set to True for less output
        )
        print("Done!\n")
        return True
    except ffmpeg.Error as e:
        print(f"Error processing {video_path}: {e.stderr.decode() if e.stderr else str(e)}\n")
        return False


def process_folder(folder_path):
    extensions = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg'}
    count = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(tuple(extensions)):
                video_path = os.path.join(root, file)
                success = video_to_audio(video_path)
                if success:
                    count += 1
    print(f"Finished! Processed {count} video files.")


# ────── RUN IT ──────
if __name__ == "__main__":
    directory = "/home/sionna/Downloads/1188976"   # Change this to your folder
    process_folder(directory)