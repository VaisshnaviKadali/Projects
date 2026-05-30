"""
Main FastAPI application for Alzheimer's Assistance System.
"""
import os
import io
import json
import base64
import numpy as np
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from pydantic import BaseModel
import bcrypt as _bcrypt

def hash_password(password: str) -> str:
    return _bcrypt.hashpw(password.encode('utf-8'), _bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return _bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

from database import init_db, get_db, Caregiver, Patient, Relative, FaceEncoding
from database import Medicine, MemoryNote, QuizSession, RecognitionLog, SOSAlert, EvaluationResult

# ──────────────────── App Setup ────────────────────

app = FastAPI(title="MindGuard — Alzheimer's Assistance System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve plots as static files
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
app.mount("/plots", StaticFiles(directory=PLOTS_DIR), name="plots")

# Lazy-loaded face recognizer
_face_recognizer = None

def get_face_recognizer():
    global _face_recognizer
    if _face_recognizer is None:
        from face_recognition_module import FaceRecognizer
        _face_recognizer = FaceRecognizer()
    return _face_recognizer

# Lazy-loaded age/gender detector
_age_gender_detector = None

def get_age_gender_detector():
    global _age_gender_detector
    if _age_gender_detector is None:
        from age_gender_module import AgeGenderDetector
        _age_gender_detector = AgeGenderDetector()
    return _age_gender_detector


@app.on_event("startup")
def startup():
    init_db()
    print("[STARTUP] Database initialized.")


# ──────────────────── Pydantic Schemas ────────────────────

class CaregiverCreate(BaseModel):
    username: str
    password: str
    email: str
    full_name: str

class CaregiverLogin(BaseModel):
    username: str
    password: str

class PatientCreate(BaseModel):
    name: str
    age: Optional[int] = None
    caregiver_id: int

class RelativeCreate(BaseModel):
    name: str
    relationship_type: str
    patient_id: int

class MedicineCreate(BaseModel):
    name: str
    dosage: Optional[str] = None
    schedule_time: str
    frequency: str = "daily"
    patient_id: int

class MedicineUpdate(BaseModel):
    name: Optional[str] = None
    dosage: Optional[str] = None
    schedule_time: Optional[str] = None
    frequency: Optional[str] = None
    is_active: Optional[bool] = None

class MemoryNoteCreate(BaseModel):
    title: str
    content: str
    category: str = "general"
    patient_id: int

class QuizSubmit(BaseModel):
    patient_id: int
    total_questions: int
    correct_answers: int
    details: Optional[str] = None

class SOSRequest(BaseModel):
    patient_id: int
    alert_type: str = "emergency"
    message: Optional[str] = None


# ──────────────────── AUTH ENDPOINTS ────────────────────

@app.post("/api/auth/register")
def register(data: CaregiverCreate, db: Session = Depends(get_db)):
    existing = db.query(Caregiver).filter(Caregiver.username == data.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")
    caregiver = Caregiver(
        username=data.username,
        password_hash=hash_password(data.password),
        email=data.email,
        full_name=data.full_name
    )
    db.add(caregiver)
    db.commit()
    db.refresh(caregiver)
    return {"id": caregiver.id, "username": caregiver.username, "message": "Registered successfully"}


@app.post("/api/auth/login")
def login(data: CaregiverLogin, db: Session = Depends(get_db)):
    caregiver = db.query(Caregiver).filter(Caregiver.username == data.username).first()
    if not caregiver or not verify_password(data.password, caregiver.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {
        "id": caregiver.id,
        "username": caregiver.username,
        "full_name": caregiver.full_name,
        "email": caregiver.email,
        "message": "Login successful"
    }


# ──────────────────── PATIENT ENDPOINTS ────────────────────

@app.post("/api/patients")
def create_patient(data: PatientCreate, db: Session = Depends(get_db)):
    patient = Patient(name=data.name, age=data.age, caregiver_id=data.caregiver_id)
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return {"id": patient.id, "name": patient.name}


@app.get("/api/patients/{caregiver_id}")
def get_patients(caregiver_id: int, db: Session = Depends(get_db)):
    patients = db.query(Patient).filter(Patient.caregiver_id == caregiver_id).all()
    return [{"id": p.id, "name": p.name, "age": p.age, "created_at": str(p.created_at)} for p in patients]


# ──────────────────── RELATIVE & FACE ENDPOINTS ────────────────────

@app.post("/api/relatives")
def create_relative(data: RelativeCreate, db: Session = Depends(get_db)):
    relative = Relative(name=data.name, relationship_type=data.relationship_type, patient_id=data.patient_id)
    db.add(relative)
    db.commit()
    db.refresh(relative)
    return {"id": relative.id, "name": relative.name}


@app.get("/api/relatives/{patient_id}")
def get_relatives(patient_id: int, db: Session = Depends(get_db)):
    relatives = db.query(Relative).filter(Relative.patient_id == patient_id).all()
    return [{"id": r.id, "name": r.name, "relationship_type": r.relationship_type,
             "encodings_count": len(r.face_encodings)} for r in relatives]


@app.post("/api/face/register")
async def register_face(
    relative_id: int = Form(...),
    condition: str = Form("normal"),
    file: UploadFile = File(...)
):
    """Register a face for a relative (supports incremental learning — multiple per person)."""
    import cv2
    recognizer = get_face_recognizer()
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces = recognizer.get_all_face_encodings(rgb_image)
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected in image")

    encoding = faces[0]["encoding"]
    db = next(get_db())
    face_enc = FaceEncoding(
        relative_id=relative_id,
        encoding=recognizer.encoding_to_bytes(encoding),
        condition=condition
    )
    db.add(face_enc)
    db.commit()
    return {"message": "Face registered", "relative_id": relative_id, "condition": condition}


@app.post("/api/face/recognize")
async def recognize_face(
    patient_id: int = Form(...),
    threshold: Optional[float] = Form(None),
    file: UploadFile = File(...)
):
    """Recognize a face from camera feed against registered relatives."""
    import cv2
    recognizer = get_face_recognizer()
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces = recognizer.get_all_face_encodings(rgb_image)
    if not faces:
        return {"recognized": False, "message": "No face detected"}

    # Load known encodings for this patient's relatives
    db = next(get_db())
    relatives = db.query(Relative).filter(Relative.patient_id == patient_id).all()
    known_map = {}
    rel_names = {}
    for rel in relatives:
        encodings = [recognizer.bytes_to_encoding(fe.encoding) for fe in rel.face_encodings]
        if encodings:
            known_map[rel.id] = encodings
            rel_names[rel.id] = {"name": rel.name, "relationship": rel.relationship_type}

    if not known_map:
        return {"recognized": False, "message": "No registered relatives found"}

    unknown_encoding = faces[0]["encoding"]
    rel_id, distance, is_known = recognizer.identify_face(unknown_encoding, known_map, threshold)

    # Log recognition
    log = RecognitionLog(
        patient_id=patient_id,
        predicted_relative_id=rel_id,
        confidence=1.0 - distance,
        threshold_used=threshold or recognizer.threshold,
        result="TP" if is_known else "Unknown",
        is_known=is_known
    )
    db.add(log)
    db.commit()

    if is_known and rel_id in rel_names:
        info = rel_names[rel_id]
        return {
            "recognized": True,
            "relative_id": rel_id,
            "name": info["name"],
            "relationship": info["relationship"],
            "confidence": round(1.0 - distance, 4),
            "distance": round(distance, 4)
        }
    return {
        "recognized": False,
        "message": "Person not recognized — may be a stranger",
        "confidence": round(1.0 - distance, 4) if distance < float("inf") else 0,
        "distance": round(distance, 4) if distance < float("inf") else None
    }


@app.post("/api/face/confirm")
async def confirm_recognition(
    relative_id: int = Form(...),
    file: UploadFile = File(...)
):
    """Caregiver confirms identity → update embeddings (incremental learning)."""
    import cv2
    recognizer = get_face_recognizer()
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces = recognizer.get_all_face_encodings(rgb_image)
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected")

    db = next(get_db())
    enc = FaceEncoding(
        relative_id=relative_id,
        encoding=recognizer.encoding_to_bytes(faces[0]["encoding"]),
        condition="incremental_update"
    )
    db.add(enc)
    db.commit()
    return {"message": "Embedding updated via incremental learning", "relative_id": relative_id}


# ──────────────────── MEDICINE ENDPOINTS ────────────────────

@app.post("/api/medicines")
def create_medicine(data: MedicineCreate, db: Session = Depends(get_db)):
    med = Medicine(**data.dict())
    db.add(med)
    db.commit()
    db.refresh(med)
    return {"id": med.id, "name": med.name, "schedule_time": med.schedule_time}


@app.get("/api/medicines/{patient_id}")
def get_medicines(patient_id: int, db: Session = Depends(get_db)):
    meds = db.query(Medicine).filter(Medicine.patient_id == patient_id, Medicine.is_active == True).all()
    return [{"id": m.id, "name": m.name, "dosage": m.dosage,
             "schedule_time": m.schedule_time, "frequency": m.frequency} for m in meds]


@app.put("/api/medicines/{medicine_id}")
def update_medicine(medicine_id: int, data: MedicineUpdate, db: Session = Depends(get_db)):
    med = db.query(Medicine).filter(Medicine.id == medicine_id).first()
    if not med:
        raise HTTPException(status_code=404, detail="Medicine not found")
    for key, value in data.dict(exclude_unset=True).items():
        setattr(med, key, value)
    db.commit()
    return {"message": "Medicine updated"}


@app.delete("/api/medicines/{medicine_id}")
def delete_medicine(medicine_id: int, db: Session = Depends(get_db)):
    med = db.query(Medicine).filter(Medicine.id == medicine_id).first()
    if not med:
        raise HTTPException(status_code=404, detail="Medicine not found")
    med.is_active = False
    db.commit()
    return {"message": "Medicine deactivated"}


# ──────────────────── MEMORY NOTES ENDPOINTS ────────────────────

@app.post("/api/memory-notes")
def create_memory_note(data: MemoryNoteCreate, db: Session = Depends(get_db)):
    note = MemoryNote(**data.dict())
    db.add(note)
    db.commit()
    db.refresh(note)
    return {"id": note.id, "title": note.title}


@app.get("/api/memory-notes/{patient_id}")
def get_memory_notes(patient_id: int, db: Session = Depends(get_db)):
    notes = db.query(MemoryNote).filter(MemoryNote.patient_id == patient_id).order_by(
        MemoryNote.created_at.desc()).all()
    return [{"id": n.id, "title": n.title, "content": n.content,
             "category": n.category, "created_at": str(n.created_at)} for n in notes]


@app.delete("/api/memory-notes/{note_id}")
def delete_memory_note(note_id: int, db: Session = Depends(get_db)):
    note = db.query(MemoryNote).filter(MemoryNote.id == note_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    db.delete(note)
    db.commit()
    return {"message": "Note deleted"}


# ──────────────────── QUIZ / MEMORY SCORE ENDPOINTS ────────────────────

@app.get("/api/quiz/generate/{patient_id}")
def generate_quiz_endpoint(patient_id: int, num_questions: int = 5, db: Session = Depends(get_db)):
    """Generate a personalized quiz for the patient."""
    from quiz_generator import generate_quiz
    questions = generate_quiz(db, patient_id, num_questions)
    return {"patient_id": patient_id, "questions": questions, "total": len(questions)}


@app.post("/api/quiz/submit")
def submit_quiz(data: QuizSubmit, db: Session = Depends(get_db)):
    from quiz_generator import get_quiz_feedback
    score = (data.correct_answers / data.total_questions) * 100 if data.total_questions > 0 else 0
    feedback = get_quiz_feedback(score)
    session = QuizSession(
        patient_id=data.patient_id,
        total_questions=data.total_questions,
        correct_answers=data.correct_answers,
        score_percentage=score,
        details=data.details
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return {"id": session.id, "score": round(score, 2), "feedback": feedback}


@app.get("/api/quiz/history/{patient_id}")
def get_quiz_history(patient_id: int, db: Session = Depends(get_db)):
    sessions = db.query(QuizSession).filter(QuizSession.patient_id == patient_id).order_by(
        QuizSession.session_date.desc()).all()
    return [{"id": s.id, "total": s.total_questions, "correct": s.correct_answers,
             "score": round(s.score_percentage, 2), "date": str(s.session_date)} for s in sessions]


# ──────────────────── SOS ALERT ENDPOINTS ────────────────────

@app.post("/api/sos/trigger")
def trigger_sos(data: SOSRequest, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == data.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    caregiver = db.query(Caregiver).filter(Caregiver.id == patient.caregiver_id).first()

    alert = SOSAlert(
        patient_id=data.patient_id,
        alert_type=data.alert_type,
        message=data.message or f"SOS Alert from patient {patient.name}!",
        email_sent=False
    )

    # Try sending email
    email_sent = False
    if caregiver and caregiver.email:
        try:
            import smtplib
            from email.mime.text import MIMEText
            msg = MIMEText(
                f"EMERGENCY ALERT!\n\nPatient: {patient.name}\n"
                f"Type: {data.alert_type}\nMessage: {data.message or 'SOS triggered'}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            msg["Subject"] = f"[LEELA SOS] Emergency Alert - {patient.name}"
            msg["From"] = "leela.alzheimer@system.local"
            msg["To"] = "leelamrutha05@gmail.com"
            # Email sending would need SMTP config — logged for now
            alert.email_sent = False
            print(f"[SOS] Alert created for patient {patient.name} → caregiver {caregiver.email}")
        except Exception as e:
            print(f"[SOS EMAIL ERROR] {e}")

    db.add(alert)
    db.commit()
    return {"message": "SOS alert triggered", "alert_id": alert.id, "email_sent": alert.email_sent}


@app.get("/api/sos/history/{patient_id}")
def get_sos_history(patient_id: int, db: Session = Depends(get_db)):
    alerts = db.query(SOSAlert).filter(SOSAlert.patient_id == patient_id).order_by(
        SOSAlert.created_at.desc()).all()
    return [{"id": a.id, "type": a.alert_type, "message": a.message,
             "email_sent": a.email_sent, "created_at": str(a.created_at)} for a in alerts]


# ──────────────────── ML EVALUATION ENDPOINTS ────────────────────

@app.post("/api/evaluation/generate-plots")
def generate_evaluation_plots():
    """Generate all ML evaluation plots and save to server/plots/."""
    from ml_evaluation import generate_all_plots
    results = generate_all_plots()
    return {"message": "All plots generated", "plots_dir": PLOTS_DIR}


@app.get("/api/evaluation/plots")
def list_plots():
    """List all available evaluation plots."""
    plots = []
    for f in sorted(os.listdir(PLOTS_DIR)):
        if f.endswith(".png"):
            plots.append({"filename": f, "url": f"/plots/{f}"})
    return plots


@app.get("/api/evaluation/summary")
def get_evaluation_summary():
    """Get the evaluation summary JSON."""
    summary_path = os.path.join(PLOTS_DIR, "evaluation_summary.json")
    if not os.path.exists(summary_path):
        raise HTTPException(status_code=404, detail="Run /api/evaluation/generate-plots first")
    with open(summary_path) as f:
        return json.load(f)


@app.get("/api/evaluation/recognition-logs/{patient_id}")
def get_recognition_logs(patient_id: int, db: Session = Depends(get_db)):
    logs = db.query(RecognitionLog).filter(RecognitionLog.patient_id == patient_id).order_by(
        RecognitionLog.created_at.desc()).all()
    return [{"id": l.id, "predicted": l.predicted_relative_id, "actual": l.actual_relative_id,
             "is_known": l.is_known, "confidence": l.confidence, "threshold": l.threshold_used,
             "result": l.result, "condition": l.condition, "created_at": str(l.created_at)} for l in logs]


# ──────────────────── AGE/GENDER DETECTION ENDPOINTS ────────────────────

@app.post("/api/age-gender/detect")
async def detect_age_gender(file: UploadFile = File(...)):
    """Detect age and gender from uploaded image."""
    import cv2
    detector = get_age_gender_detector()
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = detector.analyze_image(image)
    if not results:
        return {"detected": False, "message": "No faces detected in image"}
    
    return {"detected": True, "faces": results}


# ──────────────────── LIVE FACE DETECTION ENDPOINT ────────────────────

@app.post("/api/face/live-detect")
async def live_face_detect(
    file: UploadFile = File(...),
    patient_id: int = Form(...),
    threshold: float = Form(0.55),
    announce: bool = Form(True),
    db: Session = Depends(get_db)
):
    """
    Live face detection: recognizes known relatives or detects strangers with age/gender.
    Returns recognition result with TTS announcement.
    """
    import cv2
    from age_gender_module import get_age_gender_detector
    from tts_module import speak_relative_identified, speak_stranger_with_details
    
    # Get patient info
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    recognizer = get_face_recognizer()
    age_gender_detector = get_age_gender_detector()
    
    # Try face recognition first
    faces = recognizer.get_all_face_encodings(rgb_image)
    
    if not faces:
        # No face detected by dlib, try age/gender detector
        age_gender_results = age_gender_detector.analyze_image(image)
        if not age_gender_results:
            return {
                "type": "no_face",
                "recognized": False,
                "message": "No face detected in frame"
            }
        # Face detected by OpenCV but not dlib - treat as stranger
        face = age_gender_results[0]
        if announce:
            speak_stranger_with_details(patient.name, face["age"], face["gender"])
        return {
            "type": "stranger",
            "recognized": False,
            "message": "Unknown person detected",
            "age": face["age"],
            "gender": face["gender"],
            "age_confidence": face["age_confidence"],
            "gender_confidence": face["gender_confidence"]
        }
    
    # Load known encodings for this patient's relatives
    relatives = db.query(Relative).filter(Relative.patient_id == patient_id).all()
    known_map = {}
    rel_info = {}
    for rel in relatives:
        encodings = [recognizer.bytes_to_encoding(fe.encoding) for fe in rel.face_encodings]
        if encodings:
            known_map[rel.id] = encodings
            rel_info[rel.id] = {"name": rel.name, "relationship": rel.relationship_type}
    
    unknown_encoding = faces[0]["encoding"]
    
    if known_map:
        rel_id, distance, is_known = recognizer.identify_face(unknown_encoding, known_map, threshold)
    else:
        rel_id, distance, is_known = None, float("inf"), False
    
    # Get age/gender for additional info
    age_gender_results = age_gender_detector.analyze_image(image)
    age_info = age_gender_results[0] if age_gender_results else None
    
    if is_known and rel_id in rel_info:
        # Known relative detected
        info = rel_info[rel_id]
        if announce:
            speak_relative_identified(patient.name, info["name"], info["relationship"])
        
        return {
            "type": "known",
            "recognized": True,
            "name": info["name"],
            "relationship": info["relationship"],
            "confidence": round(1.0 - distance, 4),
            "relative_id": rel_id,
            "age_gender": age_info
        }
    else:
        # Unknown person - announce as stranger
        if age_info and announce:
            speak_stranger_with_details(patient.name, age_info["age"], age_info["gender"])
        
        return {
            "type": "stranger",
            "recognized": False,
            "message": "Unknown person detected",
            "age": age_info["age"] if age_info else "unknown",
            "gender": age_info["gender"] if age_info else "unknown",
            "age_confidence": age_info["age_confidence"] if age_info else 0,
            "gender_confidence": age_info["gender_confidence"] if age_info else 0
        }


# ──────────────────── TTS ENDPOINTS ────────────────────

class TTSRequest(BaseModel):
    text: str

class TTSAnnouncementRequest(BaseModel):
    key: str

class TTSRelativeRequest(BaseModel):
    patient_name: str
    relative_name: str
    relationship: str

class TTSMedicineRequest(BaseModel):
    patient_name: str
    medicine_name: str
    dosage: Optional[str] = None

@app.post("/api/tts/speak")
def tts_speak(data: TTSRequest):
    """Speak arbitrary text."""
    from tts_module import speak
    speak(data.text)
    return {"message": "Speech queued", "text": data.text}

@app.post("/api/tts/announcement")
def tts_announcement(data: TTSAnnouncementRequest):
    """Speak a pre-built announcement."""
    from tts_module import speak_announcement, ANNOUNCEMENTS
    if data.key not in ANNOUNCEMENTS:
        raise HTTPException(status_code=400, detail=f"Unknown announcement key. Available: {list(ANNOUNCEMENTS.keys())}")
    speak_announcement(data.key)
    return {"message": "Announcement queued", "key": data.key}

@app.post("/api/tts/relative-identified")
def tts_relative_identified(data: TTSRelativeRequest):
    """Announce when a relative is identified."""
    from tts_module import speak_relative_identified
    speak_relative_identified(data.patient_name, data.relative_name, data.relationship)
    return {"message": "Relative announcement queued"}

@app.post("/api/tts/medicine-reminder")
def tts_medicine_reminder(data: TTSMedicineRequest):
    """Speak a medicine reminder."""
    from tts_module import speak_medicine_reminder
    speak_medicine_reminder(data.patient_name, data.medicine_name, data.dosage)
    return {"message": "Medicine reminder queued"}

@app.post("/api/tts/stranger-alert")
def tts_stranger_alert(patient_name: str = Form(...)):
    """Alert when an unknown person is detected."""
    from tts_module import speak_stranger_alert
    speak_stranger_alert(patient_name)
    return {"message": "Stranger alert queued"}

@app.post("/api/tts/sos")
def tts_sos():
    """Announce SOS triggered."""
    from tts_module import speak_sos_triggered
    speak_sos_triggered()
    return {"message": "SOS announcement queued"}

@app.get("/api/tts/voices")
def get_tts_voices():
    """List available TTS voices."""
    from tts_module import get_available_voices
    return get_available_voices()


# ──────────────────── HEALTH CHECK ────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "system": "MindGuard — Alzheimer's Assistance System", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
