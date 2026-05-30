"""
Personalized Quiz Generator for Alzheimer's Assistance System.
Generates questions from patient's actual data: relatives, memory notes, medicines.
"""
import random
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from database import Patient, Relative, MemoryNote, Medicine


def generate_relative_questions(db: Session, patient_id: int, count: int = 2) -> List[Dict]:
    """Generate 'Who is this?' questions from registered relatives."""
    relatives = db.query(Relative).filter(Relative.patient_id == patient_id).all()
    if len(relatives) < 2:
        return []
    
    questions = []
    sampled = random.sample(relatives, min(count, len(relatives)))
    
    for rel in sampled:
        # Get other relatives as wrong options
        others = [r for r in relatives if r.id != rel.id]
        wrong_names = [r.name for r in random.sample(others, min(3, len(others)))]
        
        # Add some generic decoys if not enough relatives
        decoys = ["I don't remember", "A stranger", "Not sure"]
        while len(wrong_names) < 3:
            wrong_names.append(decoys.pop(0))
        
        options = [rel.name] + wrong_names[:3]
        random.shuffle(options)
        
        questions.append({
            "type": "relative",
            "question": f"Who is your {rel.relationship_type.lower()}?",
            "options": options,
            "answer": options.index(rel.name),
            "hint": f"Think about your {rel.relationship_type.lower()}...",
            "relative_id": rel.id
        })
    
    return questions


def generate_memory_note_questions(db: Session, patient_id: int, count: int = 2) -> List[Dict]:
    """Generate questions from memory notes."""
    notes = db.query(MemoryNote).filter(MemoryNote.patient_id == patient_id).all()
    if not notes:
        return []
    
    questions = []
    sampled = random.sample(notes, min(count, len(notes)))
    
    for note in sampled:
        # Create a question based on the note
        q_templates = [
            f"What do you remember about: {note.title}?",
            f"Can you recall details about '{note.title}'?",
        ]
        
        # The "correct" answer is acknowledging the memory
        options = [
            f"Yes, it's about: {note.content[:50]}..." if len(note.content) > 50 else f"Yes: {note.content}",
            "I don't remember this",
            "This doesn't seem familiar",
            "I'm not sure about this"
        ]
        
        questions.append({
            "type": "memory_note",
            "question": random.choice(q_templates),
            "options": options,
            "answer": 0,
            "hint": f"This is in your {note.category} memories...",
            "note_id": note.id
        })
    
    return questions


def generate_medicine_questions(db: Session, patient_id: int, count: int = 2) -> List[Dict]:
    """Generate questions about medicines."""
    medicines = db.query(Medicine).filter(
        Medicine.patient_id == patient_id,
        Medicine.is_active == True
    ).all()
    
    if not medicines:
        return []
    
    questions = []
    sampled = random.sample(medicines, min(count, len(medicines)))
    
    for med in sampled:
        # Question about medicine time
        correct_time = med.schedule_time
        wrong_times = ["08:00", "12:00", "18:00", "21:00", "06:00", "14:00", "16:00", "22:00"]
        wrong_times = [t for t in wrong_times if t != correct_time]
        
        options = [correct_time] + random.sample(wrong_times, 3)
        random.shuffle(options)
        
        questions.append({
            "type": "medicine",
            "question": f"What time do you take your {med.name}?",
            "options": options,
            "answer": options.index(correct_time),
            "hint": f"Think about your {med.frequency} medicine routine...",
            "medicine_id": med.id
        })
    
    return questions


def generate_relationship_questions(db: Session, patient_id: int, count: int = 1) -> List[Dict]:
    """Generate questions about relationship types."""
    relatives = db.query(Relative).filter(Relative.patient_id == patient_id).all()
    if not relatives:
        return []
    
    questions = []
    sampled = random.sample(relatives, min(count, len(relatives)))
    
    for rel in sampled:
        correct_rel = rel.relationship_type
        wrong_rels = ["Son", "Daughter", "Spouse", "Sibling", "Friend", "Neighbor", "Doctor", "Nurse"]
        wrong_rels = [r for r in wrong_rels if r.lower() != correct_rel.lower()]
        
        options = [correct_rel] + random.sample(wrong_rels, 3)
        random.shuffle(options)
        
        questions.append({
            "type": "relationship",
            "question": f"What is {rel.name}'s relationship to you?",
            "options": options,
            "answer": options.index(correct_rel),
            "hint": "Think about your family...",
            "relative_id": rel.id
        })
    
    return questions


def generate_quiz(db: Session, patient_id: int, num_questions: int = 5) -> List[Dict]:
    """Generate a full personalized quiz for a patient."""
    all_questions = []
    
    # Gather questions from all sources
    all_questions.extend(generate_relative_questions(db, patient_id, 2))
    all_questions.extend(generate_memory_note_questions(db, patient_id, 2))
    all_questions.extend(generate_medicine_questions(db, patient_id, 2))
    all_questions.extend(generate_relationship_questions(db, patient_id, 2))
    
    # If not enough personalized questions, add some fallback cognitive questions
    if len(all_questions) < num_questions:
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        caregiver_name = patient.caregiver.full_name if patient and patient.caregiver else "your caregiver"
        
        fallback = [
            {
                "type": "cognitive",
                "question": "What season are we currently in?",
                "options": ["Winter", "Spring", "Summer", "Autumn"],
                "answer": 2,  # Adjust based on actual date
                "hint": "Look outside or think about the weather..."
            },
            {
                "type": "cognitive", 
                "question": "Is it morning, afternoon, or evening right now?",
                "options": ["Morning", "Afternoon", "Evening", "Night"],
                "answer": 0,
                "hint": "Think about what you've done today..."
            },
            {
                "type": "cognitive",
                "question": f"Who helps take care of you?",
                "options": [caregiver_name, "I don't know", "Nobody", "I take care of myself"],
                "answer": 0,
                "hint": "Think about who helps you with medicines..."
            }
        ]
        all_questions.extend(fallback)
    
    # Shuffle and limit
    random.shuffle(all_questions)
    return all_questions[:num_questions]


def get_quiz_feedback(score_percentage: float) -> Dict:
    """Generate feedback based on quiz score."""
    if score_percentage >= 80:
        return {
            "level": "excellent",
            "message": "Excellent! Your memory is doing very well today.",
            "emoji": "🌟",
            "tts": "Excellent work! Your memory is very sharp today."
        }
    elif score_percentage >= 60:
        return {
            "level": "good",
            "message": "Good job! You remembered most things correctly.",
            "emoji": "👍",
            "tts": "Good job! You're doing well."
        }
    elif score_percentage >= 40:
        return {
            "level": "moderate",
            "message": "That's okay. Let's practice together more often.",
            "emoji": "💪",
            "tts": "That's okay. Practice makes perfect."
        }
    else:
        return {
            "level": "needs_help",
            "message": "Don't worry. Your caregiver is here to help you remember.",
            "emoji": "🤗",
            "tts": "Don't worry. We're here to help you."
        }
