/**
 * MindGuard — Alzheimer's Assistance System — Main App Logic
 */

// ═══════ STATE ═══════
let currentUser = null;
let currentPatientId = null;
let patients = [];

// ═══════ HELPERS ═══════
function $(id) {
  return document.getElementById(id);
}
function $$(sel) {
  return document.querySelectorAll(sel);
}

function showScreen(id) {
  $$(".screen").forEach((s) => s.classList.remove("active"));
  $(id).classList.add("active");
}

function showPage(name) {
  $$(".page").forEach((p) => p.classList.remove("active"));
  $(`page-${name}`).classList.add("active");
  $$(".nav-item").forEach((n) => n.classList.remove("active"));
  document.querySelector(`[data-page="${name}"]`).classList.add("active");
  onPageLoad(name);
}

function toast(msg, type = "info") {
  const el = $("toast");
  el.textContent = msg;
  el.className = `toast ${type}`;
  el.classList.remove("hidden");
  setTimeout(() => el.classList.add("hidden"), 3000);
}

function populateSelect(selectId, items, valueKey = "id", labelKey = "name") {
  const sel = $(selectId);
  if (!sel) return;
  sel.innerHTML = items
    .map((i) => `<option value="${i[valueKey]}">${i[labelKey]}</option>`)
    .join("");
}

function populateAllPatientSelects() {
  const selects = [
    "fr-patient-select",
    "fr-recognize-patient",
    "rel-patient-select",
    "med-patient-select",
    "mn-patient-select",
    "quiz-patient-select",
    "sos-patient-select",
    "live-patient-select",
  ];
  selects.forEach((id) => populateSelect(id, patients));
}

// ═══════ AUTH ═══════
$("login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    const res = await api.login({
      username: $("login-username").value,
      password: $("login-password").value,
    });
    console.log("[DEBUG] Login response:", res);
    if (!res || !res.id) {
      throw new Error("Invalid login response from server");
    }
    currentUser = res;
    toast("Login successful!", "success");
    enterApp();
  } catch (err) {
    toast(err.message, "error");
  }
});

$("register-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    await api.register({
      username: $("reg-username").value,
      password: $("reg-password").value,
      email: $("reg-email").value,
      full_name: $("reg-fullname").value,
    });
    toast("Account created! Please sign in.", "success");
    $("register-form").classList.add("hidden");
    $("login-form").classList.remove("hidden");
  } catch (err) {
    toast(err.message, "error");
  }
});

$("show-register").addEventListener("click", (e) => {
  e.preventDefault();
  $("login-form").classList.add("hidden");
  $("register-form").classList.remove("hidden");
});

$("show-login").addEventListener("click", (e) => {
  e.preventDefault();
  $("register-form").classList.add("hidden");
  $("login-form").classList.remove("hidden");
});

$("logout-btn").addEventListener("click", () => {
  currentUser = null;
  stopMedicineReminders();
  showScreen("login-screen");
  toast("Logged out", "info");
});

// ═══════ ENTER APP ═══════
async function enterApp() {
  console.log("[DEBUG] enterApp called, currentUser:", currentUser);
  showScreen("app-screen");
  console.log(
    "[DEBUG] After showScreen, app-screen active:",
    $("app-screen").classList.contains("active")
  );
  $("welcome-msg").textContent = `Welcome back, ${currentUser.full_name}!`;
  await loadPatients();
  showPage("dashboard");
  console.log("[DEBUG] After showPage dashboard");
  // Start medicine reminder service
  startMedicineReminders();
}

// ═══════ LOAD PATIENTS ═══════
async function loadPatients() {
  try {
    patients = await api.getPatients(currentUser.id);
    populateAllPatientSelects();
    $("stat-patients").textContent = patients.length;
  } catch {
    patients = [];
  }
}

// ═══════ PAGE LOAD HANDLERS ═══════
async function onPageLoad(page) {
  switch (page) {
    case "dashboard":
      await loadDashboard();
      break;
    case "patients":
      await renderPatients();
      break;
    case "medicines":
      await loadMedicines();
      break;
    case "memory-notes":
      await loadNotes();
      break;
    case "quiz":
      await loadQuizHistory();
      break;
    case "sos":
      await loadSOSHistory();
      break;
    case "face-recognition":
      await loadFaceRecRelatives();
      break;
    case "analytics":
      await loadAnalytics();
      break;
  }
}

// ═══════ DASHBOARD ═══════
async function loadDashboard() {
  try {
    const health = await api.health();
    $(
      "system-status"
    ).innerHTML = `<span class="status-ok">Server connected — ${health.system} v${health.version}</span>`;
  } catch {
    $(
      "system-status"
    ).innerHTML = `<span class="status-err">Server offline — start the Python server on port 8000</span>`;
  }

  $("stat-patients").textContent = patients.length;
  let totalRel = 0,
    totalMed = 0,
    totalAlerts = 0;
  for (const p of patients) {
    try {
      const rels = await api.getRelatives(p.id);
      totalRel += rels.length;
      const meds = await api.getMedicines(p.id);
      totalMed += meds.length;
      const alerts = await api.getSOSHistory(p.id);
      totalAlerts += alerts.length;
    } catch {}
  }
  $("stat-relatives").textContent = totalRel;
  $("stat-medicines").textContent = totalMed;
  $("stat-alerts").textContent = totalAlerts;
}

// ═══════ PATIENTS ═══════
$("add-patient-btn").addEventListener("click", () => {
  $("add-patient-form").classList.toggle("hidden");
});

$("save-patient-btn").addEventListener("click", async () => {
  const name = $("patient-name").value.trim();
  const age = parseInt($("patient-age").value) || null;
  if (!name) return toast("Name is required", "error");
  try {
    await api.createPatient({ name, age, caregiver_id: currentUser.id });
    toast("Patient added!", "success");
    $("patient-name").value = "";
    $("patient-age").value = "";
    $("add-patient-form").classList.add("hidden");
    await loadPatients();
    await renderPatients();
  } catch (err) {
    toast(err.message, "error");
  }
});

async function renderPatients() {
  const list = $("patients-list");
  if (!patients.length) {
    list.innerHTML =
      '<div class="card"><p class="text-muted">No patients added yet.</p></div>';
    return;
  }
  list.innerHTML = patients
    .map(
      (p) => `
    <div class="list-item">
      <div class="list-item-info">
        <h4>${p.name}</h4>
        <p>Age: ${p.age || "N/A"} | Added: ${new Date(
        p.created_at
      ).toLocaleDateString()}</p>
      </div>
      <span class="badge badge-blue">Patient #${p.id}</span>
    </div>
  `
    )
    .join("");
}

// ═══════ FACE RECOGNITION ═══════
async function loadFaceRecRelatives() {
  const patientId = $("fr-patient-select").value;
  if (patientId) {
    try {
      const rels = await api.getRelatives(patientId);
      populateSelect("fr-relative-select", rels);
    } catch {}
  }
}

$("fr-patient-select").addEventListener("change", loadFaceRecRelatives);

$("register-face-btn").addEventListener("click", async () => {
  const file = $("fr-register-file").files[0];
  if (!file) return toast("Please select an image", "error");
  const formData = new FormData();
  formData.append("relative_id", $("fr-relative-select").value);
  formData.append("condition", $("fr-condition").value);
  formData.append("file", file);
  try {
    const res = await api.registerFace(formData);
    $("register-face-result").className = "result-box mt-10 success";
    $("register-face-result").textContent = res.message;
    toast("Face registered!", "success");
  } catch (err) {
    $("register-face-result").className = "result-box mt-10 error";
    $("register-face-result").textContent = err.message;
  }
});

$("fr-threshold").addEventListener("input", (e) => {
  $("threshold-value").textContent = e.target.value;
});

$("recognize-face-btn").addEventListener("click", async () => {
  const file = $("fr-recognize-file").files[0];
  if (!file) return toast("Please select an image", "error");
  const formData = new FormData();
  formData.append("patient_id", $("fr-recognize-patient").value);
  formData.append("threshold", $("fr-threshold").value);
  formData.append("file", file);
  try {
    const res = await api.recognizeFace(formData);
    const box = $("recognize-result");
    if (res.recognized) {
      box.className = "result-box mt-10 success";
      box.innerHTML = `<strong>Identified: ${res.name}</strong> (${
        res.relationship
      })<br>Confidence: ${(res.confidence * 100).toFixed(1)}%`;
      // Voice announcement for recognized relative
      try {
        // Get patient name for TTS
        const pid = $("fr-recognize-patient").value;
        const patient = patients.find((p) => p.id == pid);
        if (patient) {
          await api.ttsRelativeIdentified({
            patient_name: patient.name,
            relative_name: res.name,
            relationship: res.relationship,
          });
        }
      } catch {}
    } else {
      box.className = "result-box mt-10 error";
      box.innerHTML = `<strong>Not recognized</strong> — ${res.message}`;
    }
  } catch (err) {
    $("recognize-result").className = "result-box mt-10 error";
    $("recognize-result").textContent = err.message;
  }
});

$("add-relative-btn").addEventListener("click", async () => {
  const name = $("rel-name").value.trim();
  if (!name) return toast("Name is required", "error");
  try {
    await api.createRelative({
      name,
      relationship_type: $("rel-type").value,
      patient_id: parseInt($("rel-patient-select").value),
    });
    toast("Relative added!", "success");
    $("rel-name").value = "";
    await loadFaceRecRelatives();
  } catch (err) {
    toast(err.message, "error");
  }
});

// ═══════ AGE/GENDER DETECTION ═══════
$("detect-age-gender-btn").addEventListener("click", async () => {
  const file = $("ag-file").files[0];
  if (!file) return toast("Please select an image", "error");
  const formData = new FormData();
  formData.append("file", file);
  try {
    const res = await api.detectAgeGender(formData);
    const box = $("age-gender-result");
    if (res.detected && res.faces.length > 0) {
      const face = res.faces[0];
      box.className = "result-box mt-10 success";
      box.innerHTML = `<strong>Detected:</strong> ${face.gender} (${(
        face.gender_confidence * 100
      ).toFixed(1)}%)<br>
        <strong>Age:</strong> ${face.age} (${(
        face.age_confidence * 100
      ).toFixed(1)}%)`;
    } else {
      box.className = "result-box mt-10 error";
      box.textContent = res.message || "No face detected";
    }
  } catch (err) {
    $("age-gender-result").className = "result-box mt-10 error";
    $("age-gender-result").textContent = err.message;
  }
});

// ═══════ LIVE WEBCAM DETECTION ═══════
let webcamStream = null;
let autoDetectInterval = null;
let isDetecting = false;

$("live-threshold").addEventListener("input", (e) => {
  $("live-threshold-value").textContent = e.target.value;
});

$("start-webcam-btn").addEventListener("click", async () => {
  try {
    const video = $("webcam-video");
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false,
    });
    video.srcObject = webcamStream;
    $("start-webcam-btn").disabled = true;
    $("stop-webcam-btn").disabled = false;
    $("detect-now-btn").disabled = false;
    toast("Camera started", "success");
  } catch (err) {
    toast("Camera access denied: " + err.message, "error");
  }
});

$("stop-webcam-btn").addEventListener("click", () => {
  stopWebcam();
  toast("Camera stopped", "info");
});

function stopWebcam() {
  if (webcamStream) {
    webcamStream.getTracks().forEach((t) => t.stop());
    webcamStream = null;
  }
  if (autoDetectInterval) {
    clearInterval(autoDetectInterval);
    autoDetectInterval = null;
  }
  $("webcam-video").srcObject = null;
  $("start-webcam-btn").disabled = false;
  $("stop-webcam-btn").disabled = true;
  $("detect-now-btn").disabled = true;
  $("auto-detect-toggle").checked = false;
  $("live-detect-result").className = "result-box mt-10";
  $("live-detect-result").innerHTML = "";
}

$("detect-now-btn").addEventListener("click", () => {
  runLiveDetection();
});

$("auto-detect-toggle").addEventListener("change", (e) => {
  if (e.target.checked) {
    toast("Auto-detection enabled (every 3s)", "info");
    runLiveDetection();
    autoDetectInterval = setInterval(runLiveDetection, 3000);
  } else {
    if (autoDetectInterval) {
      clearInterval(autoDetectInterval);
      autoDetectInterval = null;
    }
    toast("Auto-detection disabled", "info");
  }
});

async function runLiveDetection() {
  if (!webcamStream || isDetecting) return;

  const pid = $("live-patient-select").value;
  if (!pid) {
    toast("Select a patient first", "error");
    return;
  }

  isDetecting = true;
  $("detect-now-btn").textContent = "Detecting...";

  try {
    // Capture frame from video
    const video = $("webcam-video");
    const canvas = $("webcam-canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to blob
    const blob = await new Promise((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", 0.9)
    );

    // Send to API
    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");
    formData.append("patient_id", pid);
    formData.append("threshold", $("live-threshold").value);
    formData.append("announce", "true");

    const res = await api.liveDetect(formData);
    const box = $("live-detect-result");

    if (res.type === "known") {
      box.className = "result-box mt-10 known";
      let html = `<strong>✓ Recognized: ${res.name}</strong> (${
        res.relationship
      })<br>
        Confidence: ${(res.confidence * 100).toFixed(1)}%`;
      if (res.age_gender) {
        html += `<br><small>${res.age_gender.gender}, ${res.age_gender.age}</small>`;
      }
      box.innerHTML = html;
    } else if (res.type === "stranger") {
      box.className = "result-box mt-10 stranger";
      box.innerHTML = `<strong>⚠ Unknown Person</strong><br>
        Gender: ${res.gender} (${(res.gender_confidence * 100).toFixed(1)}%)<br>
        Age: ${res.age} (${(res.age_confidence * 100).toFixed(1)}%)`;
    } else {
      box.className = "result-box mt-10 no-face";
      box.innerHTML = `<em>No face detected in frame</em>`;
    }
  } catch (err) {
    $("live-detect-result").className = "result-box mt-10 error";
    $("live-detect-result").textContent = "Detection failed: " + err.message;
  }

  isDetecting = false;
  $("detect-now-btn").textContent = "🔍 Detect Now";
}

// ═══════ MEDICINES ═══════
$("add-med-btn").addEventListener("click", async () => {
  const name = $("med-name").value.trim();
  const time = $("med-time").value;
  if (!name || !time) return toast("Name and time required", "error");
  try {
    await api.createMedicine({
      name,
      dosage: $("med-dosage").value,
      schedule_time: time,
      frequency: $("med-frequency").value,
      patient_id: parseInt($("med-patient-select").value),
    });
    toast("Medicine added!", "success");
    $("med-name").value = "";
    $("med-dosage").value = "";
    await loadMedicines();
  } catch (err) {
    toast(err.message, "error");
  }
});

async function loadMedicines() {
  const pid = $("med-patient-select").value;
  if (!pid) return;
  try {
    const meds = await api.getMedicines(pid);
    $("medicines-list").innerHTML = meds.length
      ? meds
          .map(
            (m) => `
      <div class="list-item">
        <div class="list-item-info">
          <h4>${m.name}</h4>
          <p>${m.dosage || "No dosage"} | ${m.schedule_time} | ${
              m.frequency
            }</p>
        </div>
        <div class="list-item-actions">
          <button class="btn btn-sm btn-danger" onclick="removeMedicine(${
            m.id
          })">Remove</button>
        </div>
      </div>
    `
          )
          .join("")
      : '<div class="card"><p class="text-muted">No medicines added.</p></div>';
  } catch {}
}

$("med-patient-select").addEventListener("change", loadMedicines);

async function removeMedicine(id) {
  try {
    await api.deleteMedicine(id);
    toast("Medicine removed", "info");
    await loadMedicines();
  } catch (err) {
    toast(err.message, "error");
  }
}

// ═══════ MEMORY NOTES ═══════
$("add-note-btn").addEventListener("click", async () => {
  const title = $("mn-title").value.trim();
  const content = $("mn-content").value.trim();
  if (!title || !content) return toast("Title and content required", "error");
  try {
    await api.createNote({
      title,
      content,
      category: $("mn-category").value,
      patient_id: parseInt($("mn-patient-select").value),
    });
    toast("Note saved!", "success");
    $("mn-title").value = "";
    $("mn-content").value = "";
    await loadNotes();
  } catch (err) {
    toast(err.message, "error");
  }
});

async function loadNotes() {
  const pid = $("mn-patient-select").value;
  if (!pid) return;
  try {
    const notes = await api.getNotes(pid);
    $("notes-list").innerHTML = notes.length
      ? notes
          .map(
            (n) => `
      <div class="list-item">
        <div class="list-item-info">
          <h4>${n.title}</h4>
          <p>${n.content.substring(0, 80)}${
              n.content.length > 80 ? "..." : ""
            }</p>
          <p><span class="badge badge-blue">${n.category}</span> ${new Date(
              n.created_at
            ).toLocaleDateString()}</p>
        </div>
        <div class="list-item-actions">
          <button class="btn btn-sm btn-danger" onclick="removeNote(${
            n.id
          })">Delete</button>
        </div>
      </div>
    `
          )
          .join("")
      : '<div class="card"><p class="text-muted">No notes yet.</p></div>';
  } catch {}
}

$("mn-patient-select").addEventListener("change", loadNotes);

async function removeNote(id) {
  try {
    await api.deleteNote(id);
    toast("Note deleted", "info");
    await loadNotes();
  } catch (err) {
    toast(err.message, "error");
  }
}

// ═══════ QUIZ ═══════
let quizState = { questions: [], current: 0, correct: 0, currentHint: "" };

$("start-quiz-btn").addEventListener("click", async () => {
  const pid = $("quiz-patient-select").value;
  if (!pid) return toast("Select a patient first", "error");

  $("start-quiz-btn").textContent = "Loading...";
  $("start-quiz-btn").disabled = true;
  $("quiz-hint-btn").disabled = true;
  $("quiz-feedback").classList.add("hidden");

  try {
    const res = await api.generateQuiz(pid, 5);
    if (!res.questions || res.questions.length === 0) {
      toast(
        "Add some relatives, notes, or medicines first to generate personalized questions!",
        "info"
      );
      $("quiz-question").textContent =
        "No personalized questions available. Add patient data first.";
      $("start-quiz-btn").textContent = "Start Quiz";
      $("start-quiz-btn").disabled = false;
      return;
    }
    quizState = {
      questions: res.questions,
      current: 0,
      correct: 0,
      currentHint: "",
    };
    $("start-quiz-btn").textContent = "Quiz In Progress...";
    renderQuizQuestion();
  } catch (err) {
    toast(err.message, "error");
    $("start-quiz-btn").textContent = "Start Quiz";
    $("start-quiz-btn").disabled = false;
  }
});

function renderQuizQuestion() {
  if (quizState.current >= quizState.questions.length) {
    finishQuiz();
    return;
  }
  const q = quizState.questions[quizState.current];
  $("quiz-question").textContent = q.question;
  $("quiz-progress").textContent = `Question ${quizState.current + 1} of ${
    quizState.questions.length
  }`;
  $("quiz-hint").textContent = "";
  quizState.currentHint = q.hint || "";
  $("quiz-hint-btn").disabled = !quizState.currentHint;

  $("quiz-options").innerHTML = q.options
    .map(
      (opt, i) =>
        `<div class="quiz-option" onclick="answerQuiz(${i})">${opt}</div>`
    )
    .join("");
}

function answerQuiz(idx) {
  const q = quizState.questions[quizState.current];
  const options = $$(".quiz-option");
  options[q.answer].classList.add("correct");
  if (idx !== q.answer) options[idx].classList.add("wrong");
  else quizState.correct++;

  options.forEach((o) => (o.style.pointerEvents = "none"));
  $("quiz-hint-btn").disabled = true;

  setTimeout(() => {
    quizState.current++;
    renderQuizQuestion();
  }, 1200);
}

$("quiz-hint-btn").addEventListener("click", async () => {
  if (quizState.currentHint) {
    $("quiz-hint").textContent = `Hint: ${quizState.currentHint}`;
    try {
      await api.ttsSpeak(quizState.currentHint);
    } catch {}
  }
});

async function finishQuiz() {
  const pid = $("quiz-patient-select").value;
  try {
    const res = await api.submitQuiz({
      patient_id: parseInt(pid),
      total_questions: quizState.questions.length,
      correct_answers: quizState.correct,
    });

    $(
      "quiz-question"
    ).textContent = `${res.feedback.emoji} Quiz Complete! Score: ${res.score}%`;
    $("quiz-options").innerHTML = "";
    $(
      "quiz-progress"
    ).textContent = `${quizState.correct}/${quizState.questions.length} correct`;
    $("quiz-hint").textContent = "";

    // Show feedback
    const fb = $("quiz-feedback");
    fb.classList.remove("hidden");
    fb.className = `result-box mt-10 ${
      res.feedback.level === "excellent" || res.feedback.level === "good"
        ? "success"
        : "info"
    }`;
    fb.textContent = res.feedback.message;

    // Speak feedback
    try {
      await api.ttsSpeak(res.feedback.tts);
    } catch {}

    toast(`Quiz submitted — Score: ${res.score}%`, "success");
  } catch (err) {
    toast(err.message, "error");
  }
  $("start-quiz-btn").textContent = "Start Quiz";
  $("start-quiz-btn").disabled = false;
  $("quiz-hint-btn").disabled = true;
  await loadQuizHistory();
}

async function loadQuizHistory() {
  const pid = $("quiz-patient-select").value;
  if (!pid) return;
  try {
    const history = await api.getQuizHistory(pid);
    $("quiz-history").innerHTML = history.length
      ? history
          .map(
            (h) => `
      <div class="list-item">
        <div class="list-item-info">
          <h4>Score: ${h.score}%</h4>
          <p>${h.correct}/${h.total} correct | ${new Date(
              h.date
            ).toLocaleDateString()}</p>
        </div>
        <span class="badge ${
          h.score >= 70
            ? "badge-green"
            : h.score >= 50
            ? "badge-orange"
            : "badge-red"
        }">${h.score}%</span>
      </div>
    `
          )
          .join("")
      : '<p class="text-muted">No quiz history yet.</p>';
  } catch {}
}

$("quiz-patient-select").addEventListener("change", loadQuizHistory);

// ═══════ SOS ═══════
$("sos-trigger-btn").addEventListener("click", async () => {
  const pid = $("sos-patient-select").value;
  if (!pid) return toast("Select a patient", "error");
  try {
    const res = await api.triggerSOS({
      patient_id: parseInt(pid),
      alert_type: $("sos-type").value,
      message: $("sos-message").value,
    });
    $("sos-result").className = "result-box mt-10 error";
    $(
      "sos-result"
    ).innerHTML = `<strong>SOS ALERT TRIGGERED!</strong> Alert ID: ${res.alert_id}`;
    toast("SOS Alert sent!", "error");
    // Voice announcement
    try {
      await api.ttsSOS();
    } catch {}
    await loadSOSHistory();
  } catch (err) {
    toast(err.message, "error");
  }
});

async function loadSOSHistory() {
  const pid = $("sos-patient-select").value;
  if (!pid) return;
  try {
    const history = await api.getSOSHistory(pid);
    $("sos-history").innerHTML = history.length
      ? history
          .map(
            (a) => `
      <div class="list-item">
        <div class="list-item-info">
          <h4>${a.type.toUpperCase()} Alert</h4>
          <p>${a.message || "No message"} | ${new Date(
              a.created_at
            ).toLocaleString()}</p>
        </div>
        <span class="badge badge-red">${
          a.email_sent ? "Email Sent" : "Logged"
        }</span>
      </div>
    `
          )
          .join("")
      : '<p class="text-muted">No alerts yet.</p>';
  } catch {}
}

$("sos-patient-select").addEventListener("change", loadSOSHistory);

// ═══════ TTS (TEXT-TO-SPEECH) ═══════
$("tts-speak-btn").addEventListener("click", async () => {
  const text = $("tts-custom-text").value.trim();
  if (!text) return toast("Enter a message to speak", "error");
  try {
    await api.ttsSpeak(text);
    toast("Speaking...", "info");
  } catch (err) {
    toast(err.message, "error");
  }
});

$("tts-welcome-btn").addEventListener("click", async () => {
  try {
    await api.ttsAnnouncement("welcome");
    toast("Playing welcome message", "info");
  } catch (err) {
    toast(err.message, "error");
  }
});

$("tts-quiz-btn").addEventListener("click", async () => {
  try {
    await api.ttsAnnouncement("quiz_start");
    toast("Playing quiz start message", "info");
  } catch (err) {
    toast(err.message, "error");
  }
});

$("tts-sos-btn").addEventListener("click", async () => {
  try {
    await api.ttsSOS();
    toast("Playing SOS announcement", "info");
  } catch (err) {
    toast(err.message, "error");
  }
});

// ═══════ ML ANALYTICS ═══════
$("generate-plots-btn").addEventListener("click", async () => {
  $("plots-loading").classList.remove("hidden");
  $("generate-plots-btn").disabled = true;
  try {
    await api.generatePlots();
    toast("All plots generated!", "success");
    await loadAnalytics();
  } catch (err) {
    toast("Failed: " + err.message, "error");
  }
  $("plots-loading").classList.add("hidden");
  $("generate-plots-btn").disabled = false;
});

async function loadAnalytics() {
  // Load plots
  try {
    const plots = await api.getPlots();
    if (plots.length) {
      $("plots-grid").innerHTML = plots
        .map(
          (p) => `
        <div class="plot-item">
          <h4>${p.filename
            .replace(/^\d+_/, "")
            .replace(/_/g, " ")
            .replace(".png", "")
            .toUpperCase()}</h4>
          <img src="http://localhost:8000${p.url}" alt="${
            p.filename
          }" loading="lazy" />
        </div>
      `
        )
        .join("");
    }
  } catch {}

  // Load summary
  try {
    const summary = await api.getSummary();
    $("evaluation-summary").classList.remove("hidden");
    const metrics = summary.metrics || {};
    $("summary-content").innerHTML = `
      <div class="stats-grid" style="grid-template-columns: repeat(3, 1fr);">
        ${Object.entries(metrics)
          .map(
            ([k, v]) => `
          <div class="stat-card">
            <h3>${typeof v === "number" ? (v * 100).toFixed(1) + "%" : v}</h3>
            <p>${k}</p>
          </div>
        `
          )
          .join("")}
      </div>
      <p class="text-muted mt-10">ROC AUC: ${
        summary.roc_auc?.toFixed(4) || "N/A"
      } | PR AUC: ${summary.pr_auc?.toFixed(4) || "N/A"}</p>
    `;
  } catch {}
}

// ═══════ NAV ═══════
$$(".nav-item").forEach((item) => {
  item.addEventListener("click", () => showPage(item.dataset.page));
});

// ═══════ MEDICINE REMINDER SCHEDULER ═══════
let reminderInterval = null;
let lastRemindedMeds = {};

async function checkMedicineReminders() {
  if (!currentUser || patients.length === 0) return;

  const now = new Date();
  const currentTime = now.toTimeString().slice(0, 5);

  for (const patient of patients) {
    try {
      const meds = await api.getMedicines(patient.id);
      for (const med of meds) {
        const scheduleTime = med.schedule_time;
        const medKey = `${med.id}-${currentTime}`;

        if (scheduleTime === currentTime && !lastRemindedMeds[medKey]) {
          console.log(
            `[REMINDER] Time to take: ${med.name} for ${patient.name}`
          );

          try {
            await api.ttsMedicineReminder({
              patient_name: patient.name,
              medicine_name: med.name,
              dosage: med.dosage || null,
            });
            toast(`Reminder: ${patient.name} - ${med.name}`, "info");
          } catch (err) {
            console.error("[REMINDER TTS ERROR]", err);
          }

          lastRemindedMeds[medKey] = true;
          setTimeout(() => {
            delete lastRemindedMeds[medKey];
          }, 120000);
        }
      }
    } catch (err) {
      console.error(
        `[REMINDER] Error checking medicines for patient ${patient.id}:`,
        err
      );
    }
  }
}

function startMedicineReminders() {
  if (reminderInterval) return;
  console.log("[REMINDER] Medicine reminder service started");
  reminderInterval = setInterval(checkMedicineReminders, 30000);
  checkMedicineReminders();
}

function stopMedicineReminders() {
  if (reminderInterval) {
    clearInterval(reminderInterval);
    reminderInterval = null;
    console.log("[REMINDER] Medicine reminder service stopped");
  }
}

// ═══════ INIT ═══════
showScreen("login-screen");
