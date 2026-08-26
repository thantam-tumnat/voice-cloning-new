// Pipeline Explorer — donor -> Thonburian F5 -> SeedVC, with per-stage playback.
'use strict';

const $ = (id) => document.getElementById(id);
const els = {
  donorSet: $('donorSet'), emotion: $('emotion'), speaker: $('speaker'),
  speed: $('speed'), speedVal: $('speedVal'),
  text: $('text'), textHint: $('textHint'),
  runBtn: $('runBtn'), status: $('status'),
  stages: $('stages'), outMeta: $('outMeta'),
  tuneGrid: $('tuneGrid'), tuneReset: $('tuneReset'),
};

let donorSets = [];           // [{id, name, emotions:[{id, transcript}]}]
let userEditedText = false;   // stop auto-filling once the user types
let tuneDefaults = {};        // {key: default value} from the server

// Thai label + one-line hint for each F5 knob, in display order.
const TUNE_META = {
  sway_sampling_coef: { label: 'ความมีชีวิตชีวา (sway)', hint: '−1.0 = มีอารมณ์/ลื่นไหล · 0 = แบนราบ' },
  cfg_strength:       { label: 'ความเข้มอารมณ์ (cfg)',   hint: 'ยิ่งสูงยิ่งยึดสไตล์ donor แรง (เสี่ยง artifact)' },
  nfe_step:           { label: 'คุณภาพ/สเตป (NFE)',       hint: 'มากขึ้น = prosody คมขึ้น แต่ช้าลง' },
  target_rms:         { label: 'นอร์มัลไลซ์ความดัง (rms)', hint: 'ค่านอร์มัลไลซ์ที่ป้อนให้ F5 (default 0.1)' },
  keep_silence:       { label: 'เก็บช่วงเงียบ donor (ms)', hint: 'สูงขึ้น = รักษาจังหวะ/การเว้นวรรคของ donor' },
  min_silence_len:    { label: 'เงียบขั้นต่ำที่ตัด (ms)',  hint: 'ตัดเฉพาะช่วงเงียบที่ยาวเกินค่านี้' },
};

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

async function loadTuning() {
  try {
    const res = await fetch('/api/pipeline/defaults').then((r) => r.json());
    tuneDefaults = res.values || {};
    const specs = res.specs || {};
    els.tuneGrid.innerHTML = '';
    Object.keys(TUNE_META).forEach((key) => {
      const meta = TUNE_META[key];
      const spec = specs[key] || {};
      const field = document.createElement('div');
      field.className = 'pl-field';
      const def = tuneDefaults[key];
      field.innerHTML =
        `<label for="tune_${key}">${meta.label}</label>` +
        `<input type="number" id="tune_${key}" data-key="${key}"` +
        ` min="${spec.min}" max="${spec.max}" step="${spec.step}"` +
        ` value="${def}" placeholder="${def}" />` +
        `<div class="pl-hint">${meta.hint} · default ${def}</div>`;
      els.tuneGrid.appendChild(field);
    });
  } catch (e) {
    els.tuneGrid.innerHTML = '<p class="pl-hint">โหลดค่าปรับแต่งไม่สำเร็จ</p>';
  }
}

// Only send knobs the user actually changed from the default; the rest stay null
// so the server applies its own default.
function collectTuning() {
  const tuning = {};
  els.tuneGrid.querySelectorAll('input[data-key]').forEach((inp) => {
    const key = inp.dataset.key;
    const raw = inp.value.trim();
    if (raw === '') return;
    const val = parseFloat(raw);
    if (Number.isNaN(val)) return;
    if (val !== parseFloat(tuneDefaults[key])) tuning[key] = val;
  });
  return Object.keys(tuning).length ? tuning : null;
}

function resetTuning() {
  els.tuneGrid.querySelectorAll('input[data-key]').forEach((inp) => {
    inp.value = tuneDefaults[inp.dataset.key];
  });
}

function renderStages(data) {
  els.outMeta.textContent =
    `อารมณ์: ${data.emotion} · donor: ${data.donor_set} · เสียง: ${data.target} · ` +
    `ข้อความ: "${data.gen_text}"`;
  els.stages.innerHTML = '';
  if (data.tuning) {
    const t = data.tuning;
    const line = document.createElement('div');
    line.className = 'pl-tuneline';
    line.textContent =
      `ค่า F5 ที่ใช้: sway ${t.sway_sampling_coef} · cfg ${t.cfg_strength} · NFE ${t.nfe_step} · ` +
      `rms ${t.target_rms} · keep_silence ${t.keep_silence}ms · min_silence ${t.min_silence_len}ms · ` +
      `speed ${data.speed}×`;
    els.stages.appendChild(line);
  }
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
    const tuning = collectTuning();
    const res = await fetch('/api/pipeline/trace', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ donor_set, emotion, speaker_id, text: text || null, speed, tuning }),
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
els.tuneReset.addEventListener('click', resetTuning);

loadOptions();
loadTuning();
