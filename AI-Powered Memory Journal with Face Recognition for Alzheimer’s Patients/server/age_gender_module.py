"""
Age and Gender Detection Module using OpenCV DNN with Caffe models.
"""
import os
import cv2
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Model paths
FACE_PROTO = os.path.join(MODELS_DIR, "opencv_face_detector.prototxt")
FACE_MODEL = os.path.join(MODELS_DIR, "opencv_face_detector_uint8.pb")
AGE_PROTO = os.path.join(MODELS_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(MODELS_DIR, "age_net.caffemodel")
GENDER_PROTO = os.path.join(MODELS_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(MODELS_DIR, "gender_net.caffemodel")

# Age buckets and gender labels
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# Mean values for preprocessing
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)


class AgeGenderDetector:
    def __init__(self):
        # Check if models exist
        for path in [FACE_PROTO, FACE_MODEL, AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model not found: {path}. Run download_age_gender_models.py first.")
        
        # Load networks
        # Face detector uses TensorFlow format (.pb), age/gender use Caffe format
        self.face_net = cv2.dnn.readNetFromTensorflow(FACE_MODEL, FACE_PROTO)
        self.age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
        self.gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
        print("[OK] Age/Gender detection models loaded.")

    def detect_faces(self, frame, conf_threshold=0.7):
        """Detect faces in frame. Returns list of face bounding boxes."""
        frame_h, frame_w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frame_w)
                y1 = int(detections[0, 0, i, 4] * frame_h)
                x2 = int(detections[0, 0, i, 5] * frame_w)
                y2 = int(detections[0, 0, i, 6] * frame_h)
                faces.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(confidence)
                })
        return faces

    def predict_age_gender(self, frame, face_bbox, padding=20):
        """Predict age and gender for a detected face."""
        x1, y1, x2, y2 = face_bbox
        h, w = frame.shape[:2]
        
        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0:
            return None, None, 0, 0
        
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        # Predict gender
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender_idx = gender_preds[0].argmax()
        gender = GENDER_LIST[gender_idx]
        gender_conf = float(gender_preds[0][gender_idx])
        
        # Predict age
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age_idx = age_preds[0].argmax()
        age = AGE_BUCKETS[age_idx]
        age_conf = float(age_preds[0][age_idx])
        
        return gender, age, gender_conf, age_conf

    def analyze_image(self, image):
        """Full pipeline: detect faces and predict age/gender for each."""
        results = []
        faces = self.detect_faces(image)
        
        for face in faces:
            bbox = face["bbox"]
            gender, age, gender_conf, age_conf = self.predict_age_gender(image, bbox)
            if gender and age:
                results.append({
                    "bbox": bbox,
                    "face_confidence": face["confidence"],
                    "gender": gender,
                    "gender_confidence": gender_conf,
                    "age": age,
                    "age_confidence": age_conf
                })
        return results

    def analyze_image_bytes(self, image_bytes):
        """Analyze image from bytes (for API usage)."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return self.analyze_image(image)


# Singleton instance
_detector = None

def get_age_gender_detector():
    global _detector
    if _detector is None:
        _detector = AgeGenderDetector()
    return _detector
