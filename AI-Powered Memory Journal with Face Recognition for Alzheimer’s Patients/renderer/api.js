/**
 * API client for communicating with the Python FastAPI backend.
 */
const API_BASE = "http://localhost:8000/api";

const api = {
  async request(method, endpoint, data = null, isFormData = false) {
    const url = `${API_BASE}${endpoint}`;
    const options = { method };

    if (data) {
      if (isFormData) {
        options.body = data;
      } else {
        options.headers = { "Content-Type": "application/json" };
        options.body = JSON.stringify(data);
      }
    }

    try {
      const res = await fetch(url, options);
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || "Request failed");
      return json;
    } catch (err) {
      console.error(`[API] ${method} ${endpoint} failed:`, err);
      throw err;
    }
  },

  // Auth
  login: (data) => api.request("POST", "/auth/login", data),
  register: (data) => api.request("POST", "/auth/register", data),

  // Patients
  createPatient: (data) => api.request("POST", "/patients", data),
  getPatients: (caregiverId) => api.request("GET", `/patients/${caregiverId}`),

  // Relatives
  createRelative: (data) => api.request("POST", "/relatives", data),
  getRelatives: (patientId) => api.request("GET", `/relatives/${patientId}`),

  // Face Recognition
  registerFace: (formData) =>
    api.request("POST", "/face/register", formData, true),
  recognizeFace: (formData) =>
    api.request("POST", "/face/recognize", formData, true),
  confirmRecognition: (formData) =>
    api.request("POST", "/face/confirm", formData, true),
  liveDetect: (formData) =>
    api.request("POST", "/face/live-detect", formData, true),

  // Medicines
  createMedicine: (data) => api.request("POST", "/medicines", data),
  getMedicines: (patientId) => api.request("GET", `/medicines/${patientId}`),
  updateMedicine: (id, data) => api.request("PUT", `/medicines/${id}`, data),
  deleteMedicine: (id) => api.request("DELETE", `/medicines/${id}`),

  // Memory Notes
  createNote: (data) => api.request("POST", "/memory-notes", data),
  getNotes: (patientId) => api.request("GET", `/memory-notes/${patientId}`),
  deleteNote: (id) => api.request("DELETE", `/memory-notes/${id}`),

  // Quiz
  generateQuiz: (patientId, numQuestions = 5) =>
    api.request(
      "GET",
      `/quiz/generate/${patientId}?num_questions=${numQuestions}`
    ),
  submitQuiz: (data) => api.request("POST", "/quiz/submit", data),
  getQuizHistory: (patientId) =>
    api.request("GET", `/quiz/history/${patientId}`),

  // SOS
  triggerSOS: (data) => api.request("POST", "/sos/trigger", data),
  getSOSHistory: (patientId) => api.request("GET", `/sos/history/${patientId}`),

  // Age/Gender Detection
  detectAgeGender: (formData) =>
    api.request("POST", "/age-gender/detect", formData, true),

  // TTS (Text-to-Speech)
  ttsSpeak: (text) => api.request("POST", "/tts/speak", { text }),
  ttsAnnouncement: (key) => api.request("POST", "/tts/announcement", { key }),
  ttsRelativeIdentified: (data) =>
    api.request("POST", "/tts/relative-identified", data),
  ttsMedicineReminder: (data) =>
    api.request("POST", "/tts/medicine-reminder", data),
  ttsSOS: () => api.request("POST", "/tts/sos"),
  ttsVoices: () => api.request("GET", "/tts/voices"),

  // Evaluation
  generatePlots: () => api.request("POST", "/evaluation/generate-plots"),
  getPlots: () => api.request("GET", "/evaluation/plots"),
  getSummary: () => api.request("GET", "/evaluation/summary"),
  getRecognitionLogs: (patientId) =>
    api.request("GET", `/evaluation/recognition-logs/${patientId}`),

  // Health
  health: () => api.request("GET", "/health"),
};
