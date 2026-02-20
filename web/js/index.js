const speakBtn = document.getElementById("speak");
const textEl = document.getElementById("textInput");
const voiceEl = document.getElementById("voice");
const voiceMetaEl = document.getElementById("voiceMeta");
const endpointEl = document.getElementById("endpoint");
const streamEl = document.getElementById("stream");
const langFilterEl = document.getElementById("langFilter");
let allVoices = [];
const seedEl = document.getElementById("seed");
const statusEl = document.getElementById("status");
const player = document.getElementById("player");
let lastPlainText = "";
const sttFileEl = document.getElementById("sttFile");
const sttModelEl = document.getElementById("sttModel");
const sttEndpointEl = document.getElementById("sttEndpoint");
const sttBtn = document.getElementById("transcribe");
const sttStatusEl = document.getElementById("sttStatus");
const sttOutputEl = document.getElementById("sttOutput");
const sttSectionEl = document.getElementById("sttSection");
const ttsSectionEl = document.getElementById("ttsSection");
const ttsCardEl = document.getElementById("ttsCard");
const sttCardEl = document.getElementById("sttCard");
const showTtsBtn = document.getElementById("showTts");
const showSttBtn = document.getElementById("showStt");
const ACTIVE_UI_KEY = "active_ui";
const playUploadBtn = document.getElementById("playUpload");
const sttPlayer = document.getElementById("sttPlayer");

function setStatus(message, isError) {
  statusEl.textContent = message || "";
  statusEl.classList.toggle("error", Boolean(isError));
}

function setSttStatus(message, isError) {
  sttStatusEl.textContent = message || "";
  sttStatusEl.classList.toggle("error", Boolean(isError));
}

function buildWordTimings(text, totalSeconds) {
  const words = text.trim().split(/\s+/);
  if (
    words.length === 0 ||
    !Number.isFinite(totalSeconds) ||
    totalSeconds <= 0
  ) {
    return [];
  }
  const weights = words.map((w) => Math.max(1, w.length));
  const totalWeight = weights.reduce((a, b) => a + b, 0);
  let t = 0;
  return words.map((w, i) => {
    const dur = (weights[i] / totalWeight) * totalSeconds;
    const start = t;
    t += dur;
    return { word: w, start, end: t };
  });
}

function renderWords(container, timings) {
  container.innerHTML = "";
  timings.forEach((t, i) => {
    const span = document.createElement("span");
    span.textContent = t.word + " ";
    span.dataset.index = i;
    container.appendChild(span);
  });
}

function highlightWord(container, timings, currentTime) {
  if (!timings.length) return;
  let idx = timings.findIndex(
    (t) => currentTime >= t.start && currentTime < t.end,
  );
  if (idx < 0) idx = timings.length - 1;
  [...container.children].forEach((el, i) => {
    el.classList.toggle("active-word", i === idx);
  });
}

function getInputText() {
  return textEl.innerText.replace(/\s+/g, " ").trim();
}

function setEditable(isEditable) {
  textEl.setAttribute("contenteditable", String(isEditable));
}

function populateLanguageFilter(voices) {
  const langs = Array.from(
    new Set(voices.map((v) => v.language_code).filter(Boolean)),
  ).sort();
  langFilterEl.innerHTML = "";
  const allOpt = document.createElement("option");
  allOpt.value = "all";
  allOpt.textContent = "All";
  langFilterEl.appendChild(allOpt);
  langs.forEach((lang) => {
    const opt = document.createElement("option");
    opt.value = lang;
    opt.textContent = lang.toUpperCase();
    langFilterEl.appendChild(opt);
  });
}

function populateVoices(voices) {
  voiceEl.innerHTML = "";
  voices.forEach((voice) => {
    const opt = document.createElement("option");
    opt.value = voice.voice_id;
    opt.textContent = voice.name || voice.voice_id;
    opt.dataset.meta = voice.description || "";
    opt.dataset.lang = voice.language_code || "";
    opt.dataset.model = voice.model_id || "";
    voiceEl.appendChild(opt);
  });
  if (voices.length > 0) {
    voiceEl.value = voices[0].voice_id;
    updateVoiceMeta();
  }
}

async function loadVoices() {
  const base = endpointEl.value.trim().replace(/\/$/, "");
  const url = `${base}/voices`;
  try {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const data = await res.json();
    allVoices = data.voices || [];
    populateLanguageFilter(allVoices);
    applyLanguageFilter();
  } catch (err) {
    allVoices = [
      {
        voice_id: "amh-default",
        name: "Amharic Default",
        language_code: "amh",
        model_id: "facebook/mms-tts-amh",
        description: "",
      },
    ];
    populateLanguageFilter(allVoices);
    applyLanguageFilter();
  }
}

function updateVoiceMeta() {
  const option = voiceEl.selectedOptions[0];
  if (!option) {
    voiceMetaEl.textContent = "";
    return;
  }
  const parts = [];
  if (option.dataset.lang) parts.push(`Lang: ${option.dataset.lang}`);
  if (option.dataset.model) parts.push(`Model: ${option.dataset.model}`);
  if (option.dataset.meta) parts.push(option.dataset.meta);
  voiceMetaEl.textContent = parts.join(" · ");
}

voiceEl.addEventListener("change", updateVoiceMeta);
endpointEl.addEventListener("change", loadVoices);
langFilterEl.addEventListener("change", applyLanguageFilter);

function applyLanguageFilter() {
  const selectedLang = langFilterEl.value || "all";
  const filtered =
    selectedLang === "all"
      ? allVoices
      : allVoices.filter((v) => v.language_code === selectedLang);
  if (filtered.length === 0 && allVoices.length > 0) {
    populateVoices(allVoices);
  } else {
    populateVoices(filtered);
  }
}

speakBtn.addEventListener("click", async () => {
  const text = getInputText();
  if (!text) {
    setStatus("Please enter some text.", true);
    return;
  }
  lastPlainText = text;

  const payload = { text };
  const seedValue = seedEl.value.trim();
  if (seedValue) {
    payload.seed = Number(seedValue);
  }

  speakBtn.disabled = true;
  setStatus("Generating audio...");

  try {
    const base = endpointEl.value.trim().replace(/\/$/, "");
    const voiceId = voiceEl.value || "amh-default";
    const isStreaming = streamEl.value === "on";
    const path = isStreaming
      ? "text-to-speech/" + voiceId + "/stream"
      : "text-to-speech/" + voiceId;
    const url = `${base}/${path}`;
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || `HTTP ${res.status}`);
    }

    const blob = await res.blob();
    const objectUrl = URL.createObjectURL(blob);
    player.src = objectUrl;
    player.onloadedmetadata = () => {
      const timings = buildWordTimings(text, player.duration);
      setEditable(false);
      renderWords(textEl, timings);
      const tick = () => {
        if (!player.paused) {
          highlightWord(textEl, timings, player.currentTime);
        }
        requestAnimationFrame(tick);
      };
      tick();
    };
    await player.play();
    setStatus("Done.");
  } catch (err) {
    setStatus(`Error: ${err.message}`, true);
    if (lastPlainText) {
      textEl.textContent = lastPlainText;
      setEditable(true);
    }
  } finally {
    speakBtn.disabled = false;
  }
});

player.addEventListener("ended", () => {
  if (lastPlainText) {
    textEl.textContent = lastPlainText;
    setEditable(true);
  }
});

loadVoices();

function setActiveUi(active) {
  const showStt = active === "stt";
  sttSectionEl.classList.toggle("hidden", !showStt);
  ttsSectionEl.classList.toggle("hidden", showStt);
  sttCardEl.classList.toggle("hidden", !showStt);
  ttsCardEl.classList.toggle("hidden", showStt);
  showTtsBtn.classList.toggle("active", !showStt);
  showSttBtn.classList.toggle("active", showStt);
  localStorage.setItem(ACTIVE_UI_KEY, active);
}

const saved = localStorage.getItem(ACTIVE_UI_KEY);
setActiveUi(saved === "stt" ? "stt" : "tts");

showTtsBtn.addEventListener("click", () => setActiveUi("tts"));
showSttBtn.addEventListener("click", () => setActiveUi("stt"));

function updateUploadedAudioPreview() {
  const file = sttFileEl.files && sttFileEl.files[0];
  if (!file) {
    sttPlayer.removeAttribute("src");
    return;
  }
  const objectUrl = URL.createObjectURL(file);
  sttPlayer.src = objectUrl;
}

sttFileEl.addEventListener("change", updateUploadedAudioPreview);

playUploadBtn.addEventListener("click", () => {
  if (!sttPlayer.src) {
    updateUploadedAudioPreview();
  }
  if (sttPlayer.src) {
    sttPlayer.play();
  } else {
    setSttStatus("Please choose an audio file to play.", true);
  }
});

sttBtn.addEventListener("click", async () => {
  const file = sttFileEl.files && sttFileEl.files[0];
  if (!file) {
    setSttStatus("Please choose an audio file.", true);
    return;
  }

  const base = sttEndpointEl.value.trim().replace(/\/$/, "");
  const modelId = sttModelEl.value.trim();
  const url = modelId
    ? `${base}/speech-to-text?model_id=${encodeURIComponent(modelId)}`
    : `${base}/speech-to-text`;

  const form = new FormData();
  form.append("audio", file);

  sttBtn.disabled = true;
  setSttStatus("Transcribing...");
  sttOutputEl.value = "";

  try {
    const res = await fetch(url, {
      method: "POST",
      body: form,
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || `HTTP ${res.status}`);
    }
    const data = await res.json();
    sttOutputEl.value = data.text || "";
    setSttStatus("Done.");
  } catch (err) {
    setSttStatus(`Error: ${err.message}`, true);
  } finally {
    sttBtn.disabled = false;
  }
});
