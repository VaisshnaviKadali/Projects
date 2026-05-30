"""
Download Caffe models for Age and Gender detection.
Models are saved to server/models/ directory.
"""
import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Age detection model URLs
AGE_MODELS = {
    "age_deploy.prototxt": "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/age_deploy.prototxt",
    "age_net.caffemodel": "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/age_net.caffemodel"
}

# Gender detection model URLs  
GENDER_MODELS = {
    "gender_deploy.prototxt": "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/gender_deploy.prototxt",
    "gender_net.caffemodel": "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/gender_net.caffemodel"
}

# Face detection (OpenCV DNN) - using res10 SSD model
FACE_MODELS = {
    "opencv_face_detector.prototxt": "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/opencv_face_detector.pbtxt",
    "opencv_face_detector_uint8.pb": "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/opencv_face_detector_uint8.pb"
}

ALL_MODELS = {**AGE_MODELS, **GENDER_MODELS, **FACE_MODELS}


def download_models():
    print("=" * 50)
    print("Downloading Age/Gender Detection Models")
    print("=" * 50)
    
    for filename, url in ALL_MODELS.items():
        filepath = os.path.join(MODELS_DIR, filename)
        if os.path.exists(filepath):
            print(f"[OK] {filename} already exists.")
            continue
        print(f"[DOWNLOADING] {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            size_kb = os.path.getsize(filepath) / 1024
            print(f"[OK] {filename} ({size_kb:.1f} KB)")
        except Exception as e:
            print(f"[ERROR] Failed to download {filename}: {e}")
            raise
    
    print("\n" + "=" * 50)
    print("All Age/Gender models downloaded to server/models/")
    print("=" * 50)


if __name__ == "__main__":
    download_models()
