/* Qti π frontend logic — vanilla JS, no build step. */

const API = ""; // same-origin, backend serves this file too

// ---------------- Theme ----------------
const themeToggle = document.getElementById("themeToggle");
const body = document.body;

function applyTheme(theme) {
  body.setAttribute("data-theme", theme);
  try { localStorage.setItem("qti_theme", theme); } catch (e) { /* ignore */ }
}
(function initTheme() {
  let saved = "dark";
  try { saved = localStorage.getItem("qti_theme") || "dark"; } catch (e) { /* ignore */ }
  applyTheme(saved);
})();
themeToggle.addEventListener("click", () => {
  const current = body.getAttribute("data-theme");
  applyTheme(current === "dark" ? "light" : "dark");
});

// ---------------- Tab navigation ----------------
const navItems = document.querySelectorAll(".nav-item");
const tabs = document.querySelectorAll(".tab");
const pageTitle = document.getElementById("pageTitle");
const pageSubtitle = document.getElementById("pageSubtitle");

const TAB_META = {
  dashboard: ["Dashboard", "Live overview of your cardiac and neuro status"],
  cardiac: ["Cardiac", "Heart-disease, stroke, and arrhythmia screening"],
  neuro: ["Neuro (Ear-EEG)", "Band-power features, seizure risk, tremor check"],
  alerts: ["Alerts", "Active advisories derived from your current risk score"],
  assistant: ["Assistant", "Ask about your vitals, readings, and risk"],
  settings: ["Settings", "Emergency contacts and about"],
};

navItems.forEach((btn) => {
  btn.addEventListener("click", () => {
    navItems.forEach((b) => b.classList.remove("active"));
    tabs.forEach((t) => t.classList.remove("active"));
    btn.classList.add("active");
    const tab = btn.dataset.tab;
    document.getElementById(`tab-${tab}`).classList.add("active");
    const meta = TAB_META[tab];
    if (meta) { pageTitle.textContent = meta[0]; pageSubtitle.textContent = meta[1]; }
    if (tab === "alerts") loadAlerts();
  });
});

// ---------------- Connection badge ----------------
const connBadge = document.getElementById("connBadge");
async function pingBackend() {
  try {
    const res = await fetch(`${API}/`);
    if (res.ok) { connBadge.textContent = "● online"; connBadge.className = "conn-badge online"; return; }
    throw new Error("bad status");
  } catch (e) {
    connBadge.textContent = "● offline"; connBadge.className = "conn-badge offline";
  }
}
pingBackend();
setInterval(pingBackend, 15000);

// ---------------- Helpers ----------------
function toast(msg) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 2400);
}

function pillClass(level) {
  const l = (level || "unknown").toLowerCase();
  if (["high", "moderate", "low", "none"].includes(l)) return l;
  return "unknown";
}

function formToJson(form) {
  const data = {};
  new FormData(form).forEach((v, k) => {
    if (v === "") return;
    const num = Number(v);
    data[k] = isNaN(num) ? v : num;
  });
  return data;
}

// ---------------- Vitals form ----------------
document.getElementById("vitalsForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = formToJson(e.target);
  try {
    await fetch(`${API}/vitals`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    toast("Vitals submitted");
    await refreshRisk();
    renderCardiacResult();
  } catch (err) { toast("Failed to submit vitals — is the backend running?"); }
});

// Static reference info for each cardiac model — shown regardless of whether
// a reading has been submitted yet, so the panel always explains what each
// model predicts and how well it performs, not just the raw live output.
const CARDIAC_MODEL_INFO = [
  {
    key: "heart_risk",
    title: "Heart Disease Risk",
    badge: "85.3% accuracy · 0.915 AUC",
    desc: "Screens for a heart-disease-like pattern from clinical vitals — age, blood pressure, and related cardiac signals.",
    dataset: "Trained on UCI Heart Failure Prediction (Cleveland + Hungary + Switzerland + VA + Statlog merge) — 918 real patient records.",
  },
  {
    key: "stroke_risk",
    title: "Stroke Risk",
    badge: "69.8% accuracy · 0.803 AUC",
    desc: "Screens for a stroke-risk pattern from age, BMI, and cardiovascular-history signals.",
    dataset: "Trained on CDC NHANES, 7 survey cycles (2005–2018) — 43,346 real participant records.",
  },
  {
    key: "arrhythmia_risk",
    title: "Arrhythmia Screen",
    badge: "94.0% accuracy (5-class)",
    desc: "Classifies the heartbeat pattern implied by your ECG-interval readings into Normal, Supraventricular, Ventricular, Fusion, or Unknown/paced.",
    dataset: "Trained on the MIT-BIH Arrhythmia Database (PhysioNet) — 109,494 real annotated heartbeats.",
  },
];

// What to actually do, keyed by risk_level. Kept generic and non-diagnostic
// on purpose — this is a screening aid, not a clinician.
const RISK_ACTION = {
  NONE: { text: "No action needed right now. Keep submitting readings to track trends over time.", cls: "" },
  LOW: { text: "Low signal detected — not urgent, but worth a mention at your next regular checkup.", cls: "" },
  MODERATE: { text: "Worth a conversation with a doctor soon. Not an emergency, but don't ignore it.", cls: "warn" },
  HIGH: { text: "Consider seeking medical attention promptly. This is a screening flag, not a diagnosis — a clinician should confirm.", cls: "danger" },
  UNKNOWN: { text: "Model unavailable, or no reading submitted yet — submit a reading above to run this screen.", cls: "muted" },
};

function riskAction(level) {
  const key = (level || "UNKNOWN").toString().toUpperCase();
  return RISK_ACTION[key] || RISK_ACTION.UNKNOWN;
}

function renderModelCard(info, result) {
  const r = result || {};
  const label = r.label || (r.risk_level ? r.risk_level : "No reading submitted yet");
  const action = riskAction(r.risk_level);
  const pct = r.confidence != null ? Math.round(r.confidence * 100) : null;
  const confidenceBlock = pct != null ? `
    <div class="model-confidence-track"><div class="model-confidence-fill" style="width:${pct}%"></div></div>
    <div class="model-confidence-caption">Model confidence in this call: ${pct}% — i.e. how certain the model is that "${label}" matches patterns from its training data. This is NOT the probability that you personally have the condition.</div>
  ` : "";
  return `
    <div class="model-card">
      <div class="model-card-head">
        <h3>${info.title}</h3>
        <span class="model-badge">${info.badge}</span>
      </div>
      <p class="model-desc">${info.desc}</p>
      <p class="model-dataset">${info.dataset}</p>
      <div class="model-result">
        <div class="model-result-top"><span class="model-result-label">${label}</span></div>
        ${confidenceBlock}
      </div>
      <div class="model-action ${action.cls}">${action.text}</div>
    </div>
  `;
}

async function renderCardiacResult() {
  const risk = await getRiskScore();
  const el = document.getElementById("cardiacResult");
  el.innerHTML = CARDIAC_MODEL_INFO.map((info) => renderModelCard(info, risk ? risk[info.key] : null)).join("");
}

// ---------------- Neuro form ----------------
document.getElementById("neuroForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = formToJson(e.target);
  try {
    const res = await fetch(`${API}/neuro-reading`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    const data = await res.json();
    toast("Neuro reading submitted");
    renderNeuroResult(data.seizure_risk);
    await refreshRisk();
  } catch (err) { toast("Failed to submit neuro reading — is the backend running?"); }
});

function renderNeuroResult(seizure) {
  const el = document.getElementById("neuroResult");
  if (!seizure) { el.innerHTML = '<p class="empty-hint">No reading submitted yet.</p>'; return; }
  const pct = seizure.confidence != null ? (seizure.confidence * 100).toFixed(0) + "%" : "n/a";
  el.innerHTML = `
    <div class="kv-row"><span class="k">Seizure risk tier</span><span class="v">${seizure.risk_level}</span></div>
    <div class="kv-row"><span class="k">Model-estimated seizure probability</span><span class="v">${pct}</span></div>
    <div class="kv-sub">This is the model's own estimated probability of seizure activity from this reading's band-power pattern — it is an estimate produced by a screening model, not a clinical measurement of your actual seizure risk. A low percentage here means the current reading did not resemble the seizure-associated patterns the model was trained on.</div>
    <div class="kv-row"><span class="k">Notes</span><span class="v">${seizure.label || ""}</span></div>
  `;
}

// ---------------- Risk score / dashboard ----------------
async function getRiskScore() {
  try {
    const loc = window.__liveLoc || { lat: null, lon: null };
    const params = [];
    if (loc.lat != null) params.push(`lat=${loc.lat}`);
    if (loc.lon != null) params.push(`lon=${loc.lon}`);
    const qs = params.length ? `?${params.join("&")}` : "";
    const res = await fetch(`${API}/risk-score${qs}`);
    if (!res.ok) return null;
    return await res.json();
  } catch (e) { return null; }
}

async function refreshRisk() {
  const risk = await getRiskScore();
  if (!risk) return;

  document.getElementById("kpiHr").innerHTML = `${risk.environment ? "" : ""}` + (risk.heart_risk ? "" : "");
  const overall = risk.overall || {};
  document.getElementById("overallScore").textContent = overall.score != null ? Math.round(overall.score) : "--";
  document.getElementById("overallLevel").textContent = overall.risk_level ? `${overall.risk_level} risk` : "Awaiting data";
  document.getElementById("overallDetail").textContent = `Computed from cardiac, neuro, and environmental signals. Score ${overall.score ?? "--"} / 100.`;

  const ring = document.getElementById("overallRing");
  const deg = Math.min(360, Math.max(0, (overall.score || 0) * 3.6));
  const color = overall.risk_level === "HIGH" ? "var(--danger)" : overall.risk_level === "MODERATE" ? "var(--warn)" : "var(--accent)";
  ring.style.background = `conic-gradient(${color} 0deg ${deg}deg, var(--surface-2) ${deg}deg 360deg)`;

  const riskList = document.getElementById("riskList");
  riskList.innerHTML = "";
  const rows = [
    ["Heart disease", risk.heart_risk],
    ["Stroke", risk.stroke_risk],
    ["Arrhythmia", risk.arrhythmia_risk],
    ["Seizure (EEG)", risk.seizure_risk],
    ["Heat stress", risk.heat_stress],
    ["Dehydration", risk.dehydration],
    ["Fatigue", risk.fatigue],
    ["Fall", risk.fall],
    ["Body temperature", risk.body_temperature],
    ["Tremor", risk.tremor],
  ];
  rows.forEach(([label, r]) => {
    if (!r) return;
    const row = document.createElement("div");
    row.className = "risk-row";
    row.innerHTML = `<span>${label}</span><span class="pill ${pillClass(r.risk_level)}">${r.risk_level || "n/a"}</span>`;
    riskList.appendChild(row);
  });

  document.getElementById("kpiSeizure").innerHTML = `${risk.seizure_risk ? risk.seizure_risk.risk_level : "--"}`;

  const env = risk.environment || {};
  const envCity = document.getElementById("envCity");
  const envCoords = document.getElementById("envCoords");
  const envTemp = document.getElementById("envTemp");
  const envHumidity = document.getElementById("envHumidity");
  const envAqi = document.getElementById("envAqi");
  const envSource = document.getElementById("envSource");
  if (envCity) envCity.textContent = env.city || "--";
  if (envCoords) envCoords.textContent = (env.lat != null && env.lon != null) ? `${env.lat.toFixed(5)}, ${env.lon.toFixed(5)}` : "--";
  if (envTemp) envTemp.textContent = env.temp_c != null ? `${env.temp_c}°C` : "--";
  if (envHumidity) envHumidity.textContent = env.humidity_pct != null ? `${env.humidity_pct}%` : "--";
  if (envAqi) envAqi.textContent = env.aqi != null ? env.aqi : "--";
  if (envSource) envSource.textContent = env.source || "--";
}

// ---------------- KPI vitals ----------------
async function refreshVitalsKpis() {
  try {
    const res = await fetch(`${API}/vitals/history?limit=1`);
    const data = await res.json();
    const latest = (data.history || [])[0];
    if (!latest) return;
    document.getElementById("kpiHr").innerHTML = `${latest.heart_rate ?? "--"}<small>bpm</small>`;
    document.getElementById("kpiSpo2").innerHTML = `${latest.spo2 ?? "--"}<small>%</small>`;
    document.getElementById("kpiTemp").innerHTML = `${latest.body_temp ?? "--"}<small>°C</small>`;
  } catch (e) { /* ignore */ }
}

// ---------------- Alerts ----------------
async function loadAlerts() {
  const el = document.getElementById("alertList");
  try {
    const loc = window.__liveLoc || { lat: null, lon: null };
    const params = [];
    if (loc.lat != null) params.push(`lat=${loc.lat}`);
    if (loc.lon != null) params.push(`lon=${loc.lon}`);
    const qs = params.length ? `?${params.join("&")}` : "";
    const res = await fetch(`${API}/alerts${qs}`);
    const data = await res.json();
    if (!data.alerts || data.alerts.length === 0) {
      el.innerHTML = '<p class="empty-hint">No active alerts.</p>';
      return;
    }
    el.innerHTML = "";
    data.alerts.forEach((a) => {
      const item = document.createElement("div");
      item.className = "alert-item";
      item.textContent = a;
      el.appendChild(item);
    });
  } catch (e) { el.innerHTML = '<p class="empty-hint">Could not load alerts — is the backend running?</p>'; }
}
document.getElementById("refreshAlerts").addEventListener("click", loadAlerts);

// ---------------- Live location tracking ----------------
window.__liveLoc = { lat: null, lon: null };
const locBadge = document.getElementById("locBadge");
const locStatusDetail = document.getElementById("locStatusDetail");
const locCoords = document.getElementById("locCoords");
const locMapsLink = document.getElementById("locMapsLink");

function updateLocationUi(lat, lon, statusText, cls) {
  if (locBadge) { locBadge.textContent = `◎ ${statusText}`; locBadge.className = `loc-badge ${cls}`; }
  if (locStatusDetail) locStatusDetail.textContent = statusText;
  if (lat != null && lon != null) {
    if (locCoords) locCoords.textContent = `${lat.toFixed(5)}, ${lon.toFixed(5)}`;
    if (locMapsLink) locMapsLink.innerHTML = `<a href="https://www.google.com/maps?q=${lat},${lon}" target="_blank" rel="noopener">Open in Google Maps</a>`;
  }
}

function refreshLiveLocation() {
  if (!navigator.geolocation) {
    updateLocationUi(null, null, "unsupported by this browser", "denied");
    return;
  }
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      window.__liveLoc = { lat: pos.coords.latitude, lon: pos.coords.longitude };
      updateLocationUi(pos.coords.latitude, pos.coords.longitude, "live", "ok");
    },
    () => {
      window.__liveLoc = { lat: null, lon: null };
      updateLocationUi(null, null, "permission denied / unavailable", "denied");
    },
    { enableHighAccuracy: true, timeout: 8000, maximumAge: 20000 }
  );
}

function getSosLocation() {
  // Resolves to {lat, lon} or {lat: null, lon: null} — never rejects, so a
  // denied/unavailable GPS fix never blocks the SOS send itself. Prefers a
  // fresh fix over the cached window.__liveLoc so the SOS location is as
  // accurate as possible at the moment of the emergency.
  return new Promise((resolve) => {
    if (!navigator.geolocation) { resolve(window.__liveLoc); return; }
    navigator.geolocation.getCurrentPosition(
      (pos) => resolve({ lat: pos.coords.latitude, lon: pos.coords.longitude }),
      () => resolve(window.__liveLoc),
      { enableHighAccuracy: true, timeout: 6000, maximumAge: 0 }
    );
  });
}

refreshLiveLocation();
setInterval(refreshLiveLocation, 30000);

// ---------------- SOS ----------------
document.getElementById("sosBtn").addEventListener("click", async () => {
  toast("Getting your location…");
  const loc = await getSosLocation();
  try {
    const res = await fetch(`${API}/sos`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reason: "Manual SOS button pressed", lat: loc.lat, lon: loc.lon }),
    });
    const data = await res.json();
    const statuses = data.email_statuses || [];
    const emailNote = statuses.length ? ` (${statuses.join("; ")})` : "";
    toast((data.narration || "SOS triggered") + emailNote);
    addChatMsg(data.narration || "SOS triggered.", "bot");
  } catch (e) { toast("SOS failed — is the backend running?"); }
});

// ---------------- Assistant chat ----------------
const chatLog = document.getElementById("chatLog");
function addChatMsg(text, who) {
  const msg = document.createElement("div");
  msg.className = `chat-msg ${who}`;
  msg.textContent = text;
  chatLog.appendChild(msg);
  chatLog.scrollTop = chatLog.scrollHeight;
}
addChatMsg("Hi, I'm the Qti π assistant. Ask me about your heart, stroke, seizure, tremor, heat, or fatigue risk.", "bot");

document.getElementById("chatForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = document.getElementById("chatInput");
  const query = input.value.trim();
  if (!query) return;
  addChatMsg(query, "user");
  input.value = "";
  try {
    const res = await fetch(`${API}/assistant/query`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, lat: window.__liveLoc.lat, lon: window.__liveLoc.lon }),
    });
    const data = await res.json();
    addChatMsg(data.response || "Sorry, I couldn't process that.", "bot");
  } catch (err) { addChatMsg("Assistant is offline — is the backend running?", "bot"); }
});

// ---------------- Emergency contacts ----------------
async function loadContacts() {
  const el = document.getElementById("contactList");
  try {
    const res = await fetch(`${API}/emergency-contacts`);
    const data = await res.json();
    el.innerHTML = "";
    if (!data.contacts || data.contacts.length === 0) {
      el.innerHTML = '<p class="empty-hint">No contacts saved yet.</p>';
      return;
    }
    data.contacts.forEach((c) => {
      const card = document.createElement("div");
      card.className = "contact-card";
      const meta = [c.phone, c.relation, c.email].filter(Boolean).join(" · ");
      card.innerHTML = `
        <div><div class="cname">${c.name}</div><div class="cmeta">${meta}</div></div>
        <button data-id="${c.id}">Remove</button>
      `;
      card.querySelector("button").addEventListener("click", async () => {
        await fetch(`${API}/emergency-contacts/${c.id}`, { method: "DELETE" });
        loadContacts();
      });
      el.appendChild(card);
    });
  } catch (e) { el.innerHTML = '<p class="empty-hint">Could not load contacts.</p>'; }
}

document.getElementById("contactForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  // NOTE: deliberately NOT using formToJson() here -- that helper auto-converts
  // any all-digit string to a JS number (so it can coerce vitals/neuro readings
  // into floats), which silently turned phone numbers into numbers and caused
  // the backend's `phone: str` schema to reject the request with 422, meaning
  // the contact was never actually saved (and SOS then had nobody to email).
  // Contact fields must always stay strings.
  const form = e.target;
  const payload = {
    name: form.name.value.trim(),
    phone: form.phone.value.trim(),
    relation: form.relation.value.trim(),
    email: form.email.value.trim(),
  };
  if (!payload.name || !payload.phone) {
    toast("Name and phone are required");
    return;
  }
  try {
    const res = await fetch(`${API}/emergency-contacts`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      toast("Failed to add contact: " + (err.detail ? JSON.stringify(err.detail) : res.status));
      return;
    }
    form.reset();
    toast("Contact added");
    loadContacts();
  } catch (err) { toast("Failed to add contact — is the backend running?"); }
});

// ---------------- Init ----------------
refreshRisk();
renderCardiacResult();
refreshVitalsKpis();
loadContacts();
setInterval(refreshVitalsKpis, 10000);

// ---------------- Live Voice Chat (Gemini Live, full duplex) ----------------
let voiceWs = null;
let voiceMicStream = null;
let voiceMicContext = null;
let voiceMicProcessor = null;
let voicePlaybackContext = null;
let voicePlaybackCursor = 0;
let voiceActive = false;
let voiceReconnectAttempts = 0;
let voiceReconnectTimer = null;
let voiceUserStopped = false;
const VOICE_MAX_RECONNECT_ATTEMPTS = 4;

function voiceWsUrl() {
  const base = API || location.origin;
  const loc = window.__liveLoc || { lat: null, lon: null };
  const params = [];
  if (loc.lat != null) params.push(`lat=${loc.lat}`);
  if (loc.lon != null) params.push(`lon=${loc.lon}`);
  const qs = params.length ? `?${params.join("&")}` : "";
  return base.replace(/^http/, "ws") + "/ws/voice" + qs;
}

function downsampleTo16k(float32Input, inputSampleRate) {
  const targetRate = 16000;
  if (inputSampleRate === targetRate) return float32Input;
  const ratio = inputSampleRate / targetRate;
  const outLength = Math.floor(float32Input.length / ratio);
  const out = new Float32Array(outLength);
  for (let i = 0; i < outLength; i++) out[i] = float32Input[Math.floor(i * ratio)];
  return out;
}

function floatTo16BitPCM(float32Array) {
  const out = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    let s = Math.max(-1, Math.min(1, float32Array[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return out;
}

function setVoiceStatus(text) {
  const el = document.getElementById("voiceStatus");
  if (el) el.textContent = text;
}

async function startVoice(isReconnect) {
  voiceUserStopped = false;
  if (!isReconnect) voiceReconnectAttempts = 0;

  if (!voiceMicStream) {
    try {
      voiceMicStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      setVoiceStatus("Microphone permission denied or unavailable.");
      return;
    }
  }

  voiceWs = new WebSocket(voiceWsUrl());
  voiceWs.binaryType = "arraybuffer";

  voiceWs.onopen = function () {
    setVoiceStatus("Connected — listening… speak in any language.");
    voiceMicContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = voiceMicContext.createMediaStreamSource(voiceMicStream);
    voiceMicProcessor = voiceMicContext.createScriptProcessor(4096, 1, 1);
    source.connect(voiceMicProcessor);
    voiceMicProcessor.connect(voiceMicContext.destination);
    voiceMicProcessor.onaudioprocess = function (e) {
      if (!voiceActive || voiceWs.readyState !== WebSocket.OPEN) return;
      const input = e.inputBuffer.getChannelData(0);
      const down = downsampleTo16k(input, voiceMicContext.sampleRate);
      const pcm16 = floatTo16BitPCM(down);
      voiceWs.send(pcm16.buffer);
    };

    voicePlaybackContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
    voicePlaybackCursor = voicePlaybackContext.currentTime;
  };

  voiceWs.onmessage = function (event) {
    if (typeof event.data === "string") {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "transcript") {
          addChatMsg(msg.text, msg.role === "user" ? "user" : "bot");
        } else if (msg.type === "error") {
          setVoiceStatus("Error: " + msg.message);
          stopVoice();
        }
      } catch (e) { /* ignore malformed control frame */ }
      return;
    }
    // Binary audio frame: PCM16 @ 24kHz from Gemini — queue for gapless playback.
    const int16 = new Int16Array(event.data);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 0x8000;
    const buffer = voicePlaybackContext.createBuffer(1, float32.length, 24000);
    buffer.copyToChannel(float32, 0);
    const src = voicePlaybackContext.createBufferSource();
    src.buffer = buffer;
    src.connect(voicePlaybackContext.destination);
    const now = voicePlaybackContext.currentTime;
    const startAt = Math.max(now, voicePlaybackCursor);
    src.start(startAt);
    voicePlaybackCursor = startAt + buffer.duration;
  };

  voiceWs.onclose = function () {
    voiceActive = false;
    updateVoiceButton();
    if (voiceUserStopped) { setVoiceStatus("Voice session ended."); return; }
    if (voiceReconnectAttempts < VOICE_MAX_RECONNECT_ATTEMPTS) {
      voiceReconnectAttempts++;
      const delayMs = Math.min(1000 * voiceReconnectAttempts, 5000);
      setVoiceStatus(`Voice connection dropped — reconnecting (${voiceReconnectAttempts}/${VOICE_MAX_RECONNECT_ATTEMPTS})…`);
      voiceReconnectTimer = setTimeout(() => { if (!voiceUserStopped) startVoice(true); }, delayMs);
    } else {
      setVoiceStatus("Voice session ended — connection kept dropping. Tap Start Voice Chat to try again.");
    }
  };

  voiceWs.onerror = function () {
    setVoiceStatus("Voice connection error — retrying…");
  };

  voiceActive = true;
  updateVoiceButton();
}

function stopVoice() {
  voiceUserStopped = true;
  voiceActive = false;
  if (voiceReconnectTimer) { clearTimeout(voiceReconnectTimer); voiceReconnectTimer = null; }
  if (voiceMicProcessor) { voiceMicProcessor.disconnect(); voiceMicProcessor = null; }
  if (voiceMicContext) { voiceMicContext.close(); voiceMicContext = null; }
  if (voiceMicStream) { voiceMicStream.getTracks().forEach((t) => t.stop()); voiceMicStream = null; }
  if (voiceWs && voiceWs.readyState === WebSocket.OPEN) voiceWs.close();
  voiceWs = null;
  updateVoiceButton();
}

function updateVoiceButton() {
  const btn = document.getElementById("voiceBtn");
  if (!btn) return;
  btn.textContent = voiceActive ? "⏹️ Stop Voice Chat" : "🎙️ Start Voice Chat";
  btn.classList.toggle("active", voiceActive);
}

document.getElementById("voiceBtn").addEventListener("click", () => {
  if (voiceActive) stopVoice(); else startVoice();
});
