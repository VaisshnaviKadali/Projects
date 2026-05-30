"""
Model download script for Alzheimer's Assistance System.
Downloads all required ML models to server/models/ directory.
"""
from face_recognition_module import download_models

if __name__ == "__main__":
    print("=" * 50)
    print("Downloading ML Models for MindGuard")
    print("=" * 50)
    download_models()
    print("\nAll models downloaded to server/models/")
