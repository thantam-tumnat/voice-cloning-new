// Pipeline Explorer — donor -> Thonburian F5 -> SeedVC, with per-stage playback.
'use strict';

const $ = (id) => document.getElementById(id);
const els = {
  donorSet: $('donorSet'), emotion: $('emotion'), speaker: $('speaker'),
  speed: $('speed'), speedVal: $('speedVal'),
  text: $('text'), textHint: $('textHint'),
  runBtn: $('runBtn'), status: $('status'),
  stages: $('stages'), outMeta: $('outMeta'),
};

let donorSets = [];           // [{id, name, emotions:[{id, transcript}]}]
let userEditedText = false;   // stop auto-filling once the user types

const EMOTION_LABELS = {
  neutral: 'Neutral (ปกติ)', happy: 'Happy (มีความสุข)', sad: 'Sad (เศร้า)',
  angry: 'Angry (โกรธ)', frustrated: 'Frustrated (หงุดหงิด)',
};

function setStatus(msg, isErr = false, busy = false) {
  els.status.className = 'pl-status' + (isErr ? ' err' : '');
  els.status.innerHTML = (busy ? '<span class="pl-spinner"></span>' : '') + (msg || '');
}

function currentEmotions() {
  const set = donorSets.find((s) => s.id === els.donorSet.value);
  return set ? set.emotions : [];
}

function currentTranscript() {
  const emo = currentEmotions().find((e) => e.id === els.emotion.value);
  return emo ? (emo.transcript || '') : '';
}

function fillEmotions() {
  const emotions = currentEmotions();
  els.emotion.innerHTML = '';
  emotions.forEach((e) => {
    const opt = document.createElement('option');
    opt.value = e.id;
    opt.textContent = EMOTION_LABELS[e.id] || e.id;
    els.emotion.appendChild(opt);
  });
  syncTranscript();
}

function syncTranscript() {
  const t = currentTranscript();
  if (t) {
    els.textHint.textContent = 'Transcript ของ donor (แก้ไขหรือพิมพ์ใหม่ได้)';
    if (!userEditedText) els.text.value = t;
  } else {
    els.textHint.textContent = 'donor ชุดนี้ไม่มี transcript — พิมพ์ข้อความเอง';
    if (!userEditedText) els.text.value = '';
  }
}

async function loadOptions() {
  setStatus('กำลังโหลดตัวเลือก...', false, true);
  try {
    const [dsRes, spRes] = await Promise.all([
      fetch('/api/pipeline/donor-sets').then((r) => r.json()),
      fetch('/api/pipeline/speakers').then((r) => r.json()),
    ]);
    donorSets = dsRes.sets || [];
    els.donorSet.innerHTML = '';
    donorSets.forEach((s) => {
      const opt = document.createElement('option');
      opt.value = s.id;
      opt.textContent = s.name;
      els.donorSet.appendChild(opt);
    });

    const speakers = spRes.speakers || [];
    els.speaker.innerHTML = '';
    if (!speakers.length) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = '(ไม่พบเสียงใน ref/)';
      els.speaker.appendChild(opt);
    }
    speakers.forEach((sp) => {
      const opt = document.createElement('option');
      opt.value = sp.id;
      opt.textContent = sp.name || sp.id;
      els.speaker.appendChild(opt);
    });

    fillEmotions();
    setStatus('');
    if (!donorSets.length) setStatus('ไม่พบชุด donor ใน ref/emotions', true);
  } catch (e) {
    setStatus('โหลดตัวเลือกไม่สำเร็จ: ' + e, true);
  }
}

function renderStages(data) {
  els.outMeta.textContent =
    `อารมณ์: ${data.emotion} · donor: ${data.donor_set} · เสียง: ${data.target} · ` +
    `ข้อความ: "${data.gen_text}"`;
  els.stages.innerHTML = '';
  (data.stages || []).forEach((st) => {
    const card = document.createElement('div');
    card.className = 'pl-stage';
    const h = document.createElement('h4');
    h.textContent = st.label;
    const hint = document.createElement('div');
    hint.className = 'pl-hint';
    hint.textContent = st.hint || '';
    const audio = document.createElement('audio');
    audio.controls = true;
    audio.preload = 'none';
    audio.src = st.url + '?t=' + Date.now();
    card.appendChild(h);
    card.appendChild(hint);
    card.appendChild(audio);
    els.stages.appendChild(card);
  });
}

async function run() {
  const donor_set = els.donorSet.value;
  const emotion = els.emotion.value;
  const speaker_id = els.speaker.value || null;
  const text = els.text.value.trim();
  const speed = parseFloat(els.speed.value);
  if (!donor_set || !emotion) { setStatus('เลือก donor และอารมณ์ก่อน', true); return; }

  els.runBtn.disabled = true;
  setStatus('กำลังรัน F5 → SeedVC (อาจใช้เวลา ~20–60 วิ)...', false, true);
  try {
    const res = await fetch('/api/pipeline/trace', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ donor_set, emotion, speaker_id, text: text || null, speed }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || ('HTTP ' + res.status));
    renderStages(data);
    setStatus('เสร็จแล้ว ✓');
  } catch (e) {
    setStatus('รันไม่สำเร็จ: ' + e.message, true);
  } finally {
    els.runBtn.disabled = false;
  }
}

// --- wiring ---------------------------------------------------------------- //
els.donorSet.addEventListener('change', () => { userEditedText = false; fillEmotions(); });
els.emotion.addEventListener('change', () => { userEditedText = false; syncTranscript(); });
els.text.addEventListener('input', () => { userEditedText = true; });
els.speed.addEventListener('input', () => { els.speedVal.textContent = (+els.speed.value).toFixed(2) + '×'; });
els.runBtn.addEventListener('click', run);

loadOptions();
