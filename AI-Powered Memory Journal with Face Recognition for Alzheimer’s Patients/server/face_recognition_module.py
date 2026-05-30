"""
Face recognition module for Alzheimer's Assistance System.
Uses dlib's HOG face detector and ResNet face encoder.
Supports incremental learning with multiple embeddings per person.
"""
import os
import numpy as np
import pickle
import urllib.request
import bz2
from pathlib import Path

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Model URLs for dlib
MODEL_URLS = {
    "shape_predictor_68_face_landmarks.dat": {
        "url": "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
        "compressed": True
    },
    "dlib_face_recognition_resnet_model_v1.dat": {
        "url": "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2",
        "compressed": True
    },
    "mmod_human_face_detector.dat": {
        "url": "http://dlib.net/files/mmod_human_face_detector.dat.bz2",
        "compressed": True
    }
}


def download_models():
    """Download required ML models to server/models directory."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    for model_name, info in MODEL_URLS.items():
        model_path = os.path.join(MODELS_DIR, model_name)
        if os.path.exists(model_path):
            print(f"[OK] {model_name} already exists.")
            continue
        print(f"[DOWNLOADING] {model_name}...")
        compressed_path = model_path + ".bz2"
        try:
            urllib.request.urlretrieve(info["url"], compressed_path)
            if info.get("compressed"):
                print(f"[EXTRACTING] {model_name}...")
                with bz2.open(compressed_path, "rb") as f_in:
                    with open(model_path, "wb") as f_out:
                        f_out.write(f_in.read())
                os.remove(compressed_path)
            print(f"[OK] {model_name} downloaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to download {model_name}: {e}")
            raise


def ensure_models():
    """Check if models exist, download if not."""
    for model_name in MODEL_URLS:
        if not os.path.exists(os.path.join(MODELS_DIR, model_name)):
            download_models()
            return
    print("[OK] All models present.")


class FaceRecognizer:
    def __init__(self):
        import dlib
        ensure_models()
        predictor_path = os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat")
        encoder_path = os.path.join(MODELS_DIR, "dlib_face_recognition_resnet_model_v1.dat")
        cnn_detector_path = os.path.join(MODELS_DIR, "mmod_human_face_detector.dat")

        # HOG-based face detector (fast)
        self.hog_detector = dlib.get_frontal_face_detector()
        # CNN-based face detector (accurate, for benchmarking)
        self.cnn_detector = dlib.cnn_face_detection_model_v1(cnn_detector_path)
        # Shape predictor for face landmarks
        self.shape_predictor = dlib.shape_predictor(predictor_path)
        # Face encoder (128-d embeddings)
        self.face_encoder = dlib.face_recognition_model_v1(encoder_path)
        # Default threshold
        self.threshold = 0.55

    def detect_faces_hog(self, image):
        """Detect faces using HOG detector. Returns list of dlib rectangles."""
        return self.hog_detector(image, 1)

    def detect_faces_cnn(self, image):
        """Detect faces using CNN detector. Returns list of dlib mmod_rectangles."""
        detections = self.cnn_detector(image, 1)
        return [d.rect for d in detections]

    def get_face_encoding(self, image, face_rect):
        """Get 128-dimensional face encoding for a detected face."""
        import dlib
        shape = self.shape_predictor(image, face_rect)
        encoding = self.face_encoder.compute_face_descriptor(image, shape)
        return np.array(encoding)

    def get_all_face_encodings(self, image, method="hog"):
        """Detect all faces and return their encodings."""
        if method == "cnn":
            faces = self.detect_faces_cnn(image)
        else:
            faces = self.detect_faces_hog(image)

        results = []
        for face in faces:
            encoding = self.get_face_encoding(image, face)
            results.append({
                "encoding": encoding,
                "bbox": {
                    "left": face.left(),
                    "top": face.top(),
                    "right": face.right(),
                    "bottom": face.bottom()
                }
            })
        return results

    def compare_faces(self, known_encoding, unknown_encoding, threshold=None):
        """Compare two face encodings. Returns (is_match, distance)."""
        if threshold is None:
            threshold = self.threshold
        distance = np.linalg.norm(known_encoding - unknown_encoding)
        return distance < threshold, float(distance)

    def identify_face(self, unknown_encoding, known_encodings_map, threshold=None):
        """
        Identify a face against a map of {relative_id: [list of encodings]}.
        Returns (relative_id, min_distance, is_known).
        Supports incremental learning with multiple embeddings per person.
        """
        if threshold is None:
            threshold = self.threshold

        best_match_id = None
        best_distance = float("inf")

        for rel_id, encodings in known_encodings_map.items():
            for enc in encodings:
                distance = np.linalg.norm(enc - unknown_encoding)
                if distance < best_distance:
                    best_distance = distance
                    best_match_id = rel_id

        is_known = best_distance < threshold
        if not is_known:
            best_match_id = None

        return best_match_id, float(best_distance), is_known

    def encoding_to_bytes(self, encoding):
        """Serialize numpy encoding to bytes for database storage."""
        return pickle.dumps(encoding)

    def bytes_to_encoding(self, data):
        """Deserialize bytes back to numpy encoding."""
        return pickle.loads(data)
