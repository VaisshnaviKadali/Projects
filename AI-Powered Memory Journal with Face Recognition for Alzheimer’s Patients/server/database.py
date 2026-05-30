"""
Database setup and models for Alzheimer's Assistance System.
"""
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

DATABASE_URL = "sqlite:///./leela.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ──────────────────── Models ────────────────────

class Caregiver(Base):
    __tablename__ = "caregivers"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    email = Column(String(200), nullable=False)
    full_name = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    patients = relationship("Patient", back_populates="caregiver")


class Patient(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    age = Column(Integer)
    caregiver_id = Column(Integer, ForeignKey("caregivers.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    caregiver = relationship("Caregiver", back_populates="patients")
    relatives = relationship("Relative", back_populates="patient")
    medicines = relationship("Medicine", back_populates="patient")
    memory_notes = relationship("MemoryNote", back_populates="patient")
    quiz_sessions = relationship("QuizSession", back_populates="patient")
    sos_alerts = relationship("SOSAlert", back_populates="patient")


class Relative(Base):
    __tablename__ = "relatives"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    relationship_type = Column(String(100), nullable=False)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    patient = relationship("Patient", back_populates="relatives")
    face_encodings = relationship("FaceEncoding", back_populates="relative")


class FaceEncoding(Base):
    __tablename__ = "face_encodings"
    id = Column(Integer, primary_key=True, index=True)
    relative_id = Column(Integer, ForeignKey("relatives.id"))
    encoding = Column(LargeBinary, nullable=False)  # numpy array stored as bytes
    condition = Column(String(100), default="normal")  # lighting, angle, etc.
    confidence = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    relative = relationship("Relative", back_populates="face_encodings")


class Medicine(Base):
    __tablename__ = "medicines"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    dosage = Column(String(100))
    schedule_time = Column(String(50), nullable=False)  # HH:MM format
    frequency = Column(String(50), default="daily")
    patient_id = Column(Integer, ForeignKey("patients.id"))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    patient = relationship("Patient", back_populates="medicines")


class MemoryNote(Base):
    __tablename__ = "memory_notes"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(300), nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String(100), default="general")
    patient_id = Column(Integer, ForeignKey("patients.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    patient = relationship("Patient", back_populates="memory_notes")


class QuizSession(Base):
    __tablename__ = "quiz_sessions"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    total_questions = Column(Integer, nullable=False)
    correct_answers = Column(Integer, nullable=False)
    score_percentage = Column(Float, nullable=False)
    session_date = Column(DateTime, default=datetime.utcnow)
    details = Column(Text)  # JSON string of Q&A details
    patient = relationship("Patient", back_populates="quiz_sessions")


class RecognitionLog(Base):
    __tablename__ = "recognition_logs"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer)
    predicted_relative_id = Column(Integer, nullable=True)
    actual_relative_id = Column(Integer, nullable=True)
    is_known = Column(Boolean, default=False)
    confidence = Column(Float)
    threshold_used = Column(Float)
    condition = Column(String(100), default="normal")
    result = Column(String(50))  # TP, FP, TN, FN
    created_at = Column(DateTime, default=datetime.utcnow)


class SOSAlert(Base):
    __tablename__ = "sos_alerts"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    alert_type = Column(String(100), default="emergency")
    message = Column(Text)
    email_sent = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    patient = relationship("Patient", back_populates="sos_alerts")


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"
    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    category = Column(String(100))  # e.g., "face_recognition", "threshold_tuning"
    condition = Column(String(100))  # e.g., "normal_light", "low_light"
    details = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
