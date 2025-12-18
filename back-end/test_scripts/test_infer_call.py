import os
import sys
import json

# Add backend path so 'app' is importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.libs.inference import infer_upload_file

class DummyUploadFile:
    def __init__(self, path):
        self.filename = os.path.basename(path)
        self.file = open(path, "rb")

if __name__ == '__main__':
    sample_video = os.path.join(os.path.dirname(__file__), '..', 'test_sample.mp4')
    sample_video = os.path.abspath(sample_video)
    print('Using sample video:', sample_video)
    upload = DummyUploadFile(sample_video)
    res = infer_upload_file(upload, chunk_size=32, window_size=5, explain=True)
    print('Response length:', len(res))
    print('Bounding boxes returned:', len(res['bounding_box']))
    print('Inference entries returned:', len(res['inference']))
    # Print first inference item (pretty)
    if res['inference']:
        print(json.dumps(res['inference'][0], indent=2))
