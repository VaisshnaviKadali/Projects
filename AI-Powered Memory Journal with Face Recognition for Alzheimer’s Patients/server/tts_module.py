"""
Text-to-Speech Module for Alzheimer's Assistance System.
Uses pyttsx3 for offline TTS (no internet required).
"""
import pyttsx3
import threading
import queue
import os

# TTS engine singleton
_engine = None
_tts_queue = queue.Queue()
_tts_thread = None
_running = False


def _init_engine():
    """Initialize TTS engine with settings optimized for elderly users."""
    engine = pyttsx3.init()
    # Slower rate for better comprehension
    engine.setProperty('rate', 140)
    # Higher volume
    engine.setProperty('volume', 1.0)
    
    # Try to set a clear voice
    voices = engine.getProperty('voices')
    # Prefer female voice (often clearer for elderly)
    for voice in voices:
        if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break
    
    return engine


def _tts_worker():
    """Background worker thread for TTS."""
    global _engine, _running
    _engine = _init_engine()
    _running = True
    
    while _running:
        try:
            text = _tts_queue.get(timeout=1.0)
            if text is None:
                break
            _engine.say(text)
            _engine.runAndWait()
            _tts_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[TTS ERROR] {e}")


def start_tts_service():
    """Start the TTS background service."""
    global _tts_thread, _running
    if _tts_thread is None or not _tts_thread.is_alive():
        _running = True
        _tts_thread = threading.Thread(target=_tts_worker, daemon=True)
        _tts_thread.start()
        print("[TTS] Service started.")


def stop_tts_service():
    """Stop the TTS background service."""
    global _running
    _running = False
    _tts_queue.put(None)


def speak(text: str):
    """Queue text for speech. Non-blocking."""
    start_tts_service()
    _tts_queue.put(text)


def speak_sync(text: str):
    """Speak text synchronously (blocking)."""
    engine = _init_engine()
    engine.say(text)
    engine.runAndWait()


def speak_alert(patient_name: str, message: str):
    """Speak an alert message formatted for patients."""
    full_message = f"Attention {patient_name}. {message}"
    speak(full_message)


def speak_medicine_reminder(patient_name: str, medicine_name: str, dosage: str = None):
    """Speak a medicine reminder."""
    if dosage:
        msg = f"{patient_name}, it's time to take your {medicine_name}. The dosage is {dosage}."
    else:
        msg = f"{patient_name}, it's time to take your {medicine_name}."
    speak(msg)


def speak_relative_identified(patient_name: str, relative_name: str, relationship: str):
    """Announce when a relative is identified."""
    msg = f"{patient_name}, this is {relative_name}, your {relationship}."
    speak(msg)


def speak_stranger_alert(patient_name: str):
    """Alert when an unknown person is detected."""
    msg = f"Attention {patient_name}. I don't recognize this person. Please be careful."
    speak(msg)


def speak_stranger_with_details(patient_name: str, age: str, gender: str):
    """Alert when an unknown person is detected with age/gender info."""
    msg = f"Attention {patient_name}. Unknown person detected. Appears to be a {gender}, approximately {age} years old. Please be careful."
    speak(msg)


def speak_sos_triggered():
    """Announce that SOS has been triggered."""
    msg = "Emergency alert has been sent. Help is on the way. Please stay calm."
    speak(msg)


def get_available_voices():
    """List available TTS voices."""
    engine = _init_engine()
    voices = engine.getProperty('voices')
    return [{"id": v.id, "name": v.name} for v in voices]


# Pre-built announcements for common scenarios
ANNOUNCEMENTS = {
    "welcome": "Welcome to MindGuard, your Alzheimer's assistance system.",
    "login_success": "Login successful. Welcome back.",
    "face_registered": "Face has been registered successfully.",
    "quiz_start": "Let's begin the memory quiz. Take your time.",
    "quiz_complete": "Quiz complete. Great job!",
    "sos_triggered": "Emergency alert has been sent. Help is on the way."
}


def speak_announcement(key: str):
    """Speak a pre-built announcement by key."""
    if key in ANNOUNCEMENTS:
        speak(ANNOUNCEMENTS[key])
