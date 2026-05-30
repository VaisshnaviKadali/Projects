AI-Powered Memory Journal with Face Recognition for Alzheimer’s Patients

Overview
This project presents an AI-based assistive system designed to support Alzheimer’s patients by helping them recognize familiar faces and recall associated memories. The system integrates computer vision, machine learning, and assistive healthcare features into a unified desktop application.

It performs real-time face recognition, provides voice-based memory recall, and includes additional modules like medicine reminders, cognitive quizzes, SOS alerts, and ML-based performance evaluation.

Key Features

- Face Recognition
  - Identifies relatives using deep learning (128-d embeddings)
  - Real-time webcam-based detection

- Age & Gender Detection
  - Predicts demographic details for unknown individuals

- Text-to-Speech (TTS)
  - Announces names, relationships, and memory notes

- Medicine Reminder
  - Alerts patients to take medication on time

- Memory Notes
  - Stores personalized notes for memory recall

- Memory Quiz
  - Evaluates cognitive performance of patients

- SOS Alerts
  - Emergency notification system for caregivers

- ML Evaluation Dashboard
  - 15+ performance plots (Accuracy, ROC, Confusion Matrix, etc.)

System Architecture

🔹 Frontend
- Electron (Desktop App)
- HTML, CSS, JavaScript

🔹 Backend
- Python (FastAPI)
- Uvicorn (Server)
- SQLAlchemy (ORM)
- Pydantic (Validation)

🔹 Database
- SQLite (Lightweight embedded DB)

Machine Learning Models

Face Recognition
- dlib ResNet-based model
- 128-dimensional embeddings
- Euclidean distance matching (threshold = 0.55)

Age & Gender Detection
- OpenCV DNN (Caffe models)
- Age buckets: (0–2) to (60–100)
- Gender: Male / Female

 Performance Metrics

| Metric              | Value   |
|--------------------|--------|
| Accuracy           | 98.0%  |
| Precision          | 97.4%  |
| Recall             | 99.3%  |
| F1-Score           | 98.3%  |
| ROC AUC            | 0.995  |
| PR AUC             | 0.995  |
| FAR                | 4.0%   |
| FRR                | 0.67%  |
| Inference Time     | 52 ms  |

Project Structure
Med-L/
│
├── client/ (Electron Frontend)
│ ├── renderer/
│ ├── assets/
│ ├── main.js
│ └── package.json
│
├── server/ (Python Backend)
│ ├── models/
│ ├── plots/
│ ├── app.py
│ ├── database.py
│ ├── face_recognition_module.py
│ ├── age_gender_module.py
│ ├── ml_evaluation.py
│ ├── quiz_generator.py
│ └── tts_module.py
│
├── requirements.txt
└── README.md

Setup Instructions

🔹 Prerequisites
- Python 3.11+
- Node.js 18+
- Webcam
🔹 Backend Setup

```bash
cd server
pip install -r requirements.txt
python download_models.py
python app.py

