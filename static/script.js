// ---------------------------------------------------------------------
// Dashboard client script
// Polls the Flask backend (app.py) for the current prediction and
// session history, and updates the summary cards + three Chart.js
// charts. No webcam frames are ever sent to or handled by this file -
// only small JSON prediction results.
// ---------------------------------------------------------------------

const CURRENT_POLL_MS = 1000;
const HISTORY_POLL_MS = 4000;

const ENGAGEMENT_COLORS = {
  Interested: "#22c55e",
  Confused: "#f59e0b",
  Bored: "#ef4444",
};

const CLASS_NAMES = window.CLASS_NAMES || [
  "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise",
];
const ENGAGEMENT_CATEGORIES = window.ENGAGEMENT_CATEGORIES || [
  "Bored", "Confused", "Interested",
];

let consecutiveErrors = 0;

// -------------------- DOM references --------------------
const el = {
  serverStatus: document.getElementById("server-status"),
  startBtn: document.getElementById("start-btn"),
  stopBtn: document.getElementById("stop-btn"),
  errorBanner: document.getElementById("error-banner"),
  currentEmotion: document.getElementById("current-emotion"),
  currentConfidence: document.getElementById("current-confidence"),
  currentEngagement: document.getElementById("current-engagement"),
  currentScore: document.getElementById("current-score"),
  sessionDuration: document.getElementById("session-duration"),
  sessionSamples: document.getElementById("session-samples"),
  sessionStatus: document.getElementById("session-status"),
};

// -------------------- Chart setup --------------------

const emotionChart = new Chart(document.getElementById("emotionChart"), {
  type: "bar",
  data: {
    labels: CLASS_NAMES.map(capitalize),
    datasets: [{
      label: "Frames observed",
      data: CLASS_NAMES.map(() => 0),
      backgroundColor: "#4f46e5",
      borderRadius: 4,
    }],
  },
  options: {
    responsive: true,
    plugins: { legend: { display: false } },
    scales: { y: { beginAtZero: true, ticks: { precision: 0 } } },
  },
});

const engagementChart = new Chart(document.getElementById("engagementChart"), {
  type: "doughnut",
  data: {
    labels: ENGAGEMENT_CATEGORIES,
    datasets: [{
      data: ENGAGEMENT_CATEGORIES.map(() => 0),
      backgroundColor: ENGAGEMENT_CATEGORIES.map((c) => ENGAGEMENT_COLORS[c] || "#9ca3af"),
    }],
  },
  options: {
    responsive: true,
    plugins: { legend: { display: false } },
  },
});

const timeChart = new Chart(document.getElementById("timeChart"), {
  type: "line",
  data: {
    labels: [],
    datasets: [{
      label: "Estimated Engagement Score",
      data: [],
      borderColor: "#4f46e5",
      backgroundColor: "rgba(79,70,229,0.1)",
      tension: 0.25,
      fill: true,
      pointRadius: 2,
    }],
  },
  options: {
    responsive: true,
    scales: {
      y: { min: 0, max: 100, title: { display: true, text: "Estimated Engagement Score" } },
      x: { title: { display: true, text: "Time" } },
    },
  },
});

buildEngagementLegend();

// -------------------- Rendering helpers --------------------

function capitalize(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function buildEngagementLegend() {
  const ul = document.getElementById("engagement-legend");
  ul.innerHTML = "";
  ENGAGEMENT_CATEGORIES.forEach((cat) => {
    const li = document.createElement("li");
    const dot = document.createElement("span");
    dot.className = "dot";
    dot.style.background = ENGAGEMENT_COLORS[cat] || "#9ca3af";
    li.appendChild(dot);
    li.appendChild(document.createTextNode(cat));
    ul.appendChild(li);
  });
}

function setServerPill(text, variant) {
  el.serverStatus.textContent = text;
  el.serverStatus.className = "pill pill-" + variant;
}

function showError(message) {
  el.errorBanner.textContent = message;
  el.errorBanner.classList.remove("hidden");
}

function clearError() {
  el.errorBanner.classList.add("hidden");
}

function engagementClass(engagement) {
  if (!engagement) return "";
  return "engagement-" + engagement.toLowerCase();
}

// -------------------- Current-state polling --------------------

async function pollCurrent() {
  try {
    const res = await fetch("/api/current");
    if (!res.ok) throw new Error("Server returned " + res.status);
    const data = await res.json();
    consecutiveErrors = 0;
    clearError();
    renderCurrent(data);
  } catch (err) {
    consecutiveErrors += 1;
    setServerPill("Server unavailable", "error");
    if (consecutiveErrors >= 3) {
      showError("Lost connection to the server. Is app.py still running?");
    }
  }
}

function renderCurrent(data) {
  // Server connection itself is fine if we got here
  if (data.status === "running" && !data.stale) {
    setServerPill("Connected", "ok");
  } else if (data.status === "running" && data.stale) {
    setServerPill("Stale data", "warn");
  } else if (data.status === "error") {
    setServerPill("Error", "error");
  } else {
    setServerPill("Idle", "muted");
  }

  el.startBtn.disabled = data.status === "running" || data.status === "starting";
  el.stopBtn.disabled = !(data.status === "running" || data.status === "starting");

  if (data.status === "error" && data.error) {
    showError(data.error);
  } else if (data.status !== "error") {
    clearError();
  }

  el.sessionDuration.textContent = formatDuration(data.session.duration_seconds);
  el.sessionSamples.textContent = data.session.samples_recorded;
  el.sessionStatus.textContent = capitalize(data.status) + (data.demo_mode ? " (demo)" : "");

  const sessionNotActive = data.status !== "running";
  const noFaceOrStale = data.status === "running" && (!data.face_detected || data.stale);

  if (sessionNotActive) {
    el.currentEmotion.textContent = "--";
    el.currentConfidence.textContent = "Waiting for student...";
    el.currentEngagement.textContent = "--";
    el.currentEngagement.className = "card-value";
    el.currentScore.textContent = "Score: --";
    return;
  }

  if (noFaceOrStale) {
    el.currentEmotion.textContent = "No face detected";
    el.currentConfidence.textContent = data.stale ? "Waiting for student..." : "";
    // Keep last-known engagement visible but don't imply it's live -
    // engagement.py already holds the last state through brief gaps.
  }

  if (data.emotion) {
    el.currentEmotion.textContent = noFaceOrStale ? "No face detected" : capitalize(data.emotion);
    el.currentConfidence.textContent =
      (data.emotion_confidence != null ? (data.emotion_confidence * 100).toFixed(1) + "%" : "--") +
      (data.reliable === false ? " (low reliability)" : "");
  }

  if (data.engagement) {
    el.currentEngagement.textContent = data.engagement;
    el.currentEngagement.className = "card-value " + engagementClass(data.engagement);
    el.currentScore.textContent = "Score: " + (data.engagement_score ?? "--");
  }
}

function formatDuration(seconds) {
  const s = Math.floor(seconds % 60);
  const m = Math.floor(seconds / 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

// -------------------- History polling + chart updates --------------------

async function pollHistory() {
  try {
    const res = await fetch("/api/history");
    if (!res.ok) throw new Error("Server returned " + res.status);
    const data = await res.json();
    renderHistory(data.history);
  } catch (err) {
    // Current-state polling already surfaces connectivity errors;
    // avoid duplicating the error banner here.
  }
}

function renderHistory(history) {
  // --- Emotion distribution ---
  const emotionCounts = Object.fromEntries(CLASS_NAMES.map((c) => [c, 0]));
  history.forEach((r) => {
    if (r.emotion in emotionCounts) emotionCounts[r.emotion] += 1;
  });
  emotionChart.data.datasets[0].data = CLASS_NAMES.map((c) => emotionCounts[c]);
  emotionChart.update();

  // --- Engagement distribution ---
  const engagementCounts = Object.fromEntries(ENGAGEMENT_CATEGORIES.map((c) => [c, 0]));
  history.forEach((r) => {
    if (r.engagement in engagementCounts) engagementCounts[r.engagement] += 1;
  });
  engagementChart.data.datasets[0].data = ENGAGEMENT_CATEGORIES.map((c) => engagementCounts[c]);
  engagementChart.update();

  // --- Engagement over time ---
  const recent = history.slice(-60); // keep the time chart readable
  timeChart.data.labels = recent.map((r) =>
    new Date(r.timestamp * 1000).toLocaleTimeString()
  );
  timeChart.data.datasets[0].data = recent.map((r) => r.engagement_score);
  timeChart.update();
}

// -------------------- Start / Stop controls --------------------

el.startBtn.addEventListener("click", async () => {
  el.startBtn.disabled = true;
  try {
    const res = await fetch("/api/start", { method: "POST" });
    const data = await res.json();
    if (!data.ok) showError(data.message);
  } catch (err) {
    showError("Could not reach the server to start the session.");
  }
  pollCurrent();
});

el.stopBtn.addEventListener("click", async () => {
  el.stopBtn.disabled = true;
  try {
    const res = await fetch("/api/stop", { method: "POST" });
    const data = await res.json();
    if (!data.ok) showError(data.message);
  } catch (err) {
    showError("Could not reach the server to stop the session.");
  }
  pollCurrent();
});

// -------------------- Kick off polling --------------------

pollCurrent();
pollHistory();
setInterval(pollCurrent, CURRENT_POLL_MS);
setInterval(pollHistory, HISTORY_POLL_MS);
