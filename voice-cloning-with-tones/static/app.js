document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements - Input Side
  const textInput = document.getElementById('text-input');
  const guidanceInput = document.getElementById('guidance-input');
  const charCounter = document.getElementById('char-counter');
  const engineSelect = document.getElementById('engine-select');
  const speakerSelect = document.getElementById('speaker-select');
  const speakerGroup = document.getElementById('speaker-group');
  const btnPlaySpeakerRef = document.getElementById('btn-play-speaker-ref');
  const speakerRefAudio = document.getElementById('speaker-ref-audio');
  const uploadVoiceArea = document.getElementById('upload-voice-area');
  const audioFileInput = document.getElementById('audio-file-input');
  const dropZone = document.getElementById('drop-zone');
  const dropZoneContent = document.getElementById('drop-zone-content');
  const fileInfoBadge = document.getElementById('file-info-badge');
  const selectedFilename = document.getElementById('selected-filename');
  const btnPlayUploadedRef = document.getElementById('btn-play-uploaded-ref');
  const uploadedRefAudio = document.getElementById('uploaded-ref-audio');
  const btnRemoveFile = document.getElementById('btn-remove-file');
  const btnSaveSpeaker = document.getElementById('btn-save-speaker');
  const btnRerollSeed = document.getElementById('btn-reroll-seed');

  // LLM Model Selector Elements
  const llmModelSelect = document.getElementById('llm-model-select');
  const customModelGroup = document.getElementById('custom-model-group');
  const customModelInput = document.getElementById('custom-model-input');

  const paramCfg = document.getElementById('param-cfg');
  const paramSteps = document.getElementById('param-steps');
  const valCfg = document.getElementById('val-cfg');
  const valSteps = document.getElementById('val-steps');
  const btnCfgMinus = document.getElementById('btn-cfg-minus');
  const btnCfgPlus = document.getElementById('btn-cfg-plus');
  const cfgPillButtons = document.querySelectorAll('.cfg-pill-btn');

  const btnProcess = document.getElementById('btn-process');
  const btnSynthesizeDirect = document.getElementById('btn-synthesize-direct');
  const btnClear = document.getElementById('btn-clear');
  const presetButtons = document.querySelectorAll('.preset-btn');
  const healthStatus = document.getElementById('health-status');
  
  // DOM Elements - Output / Editor & Audio Player Side
  const modelBadge = document.getElementById('model-badge');
  const modelName = document.getElementById('model-name');
  const fallbackIndicator = document.getElementById('fallback-indicator');

  // Error Banner Elements
  const errorBanner = document.getElementById('error-banner');
  const errorTitle = document.getElementById('error-title');
  const errorDetailText = document.getElementById('error-detail-text');
  const btnCloseError = document.getElementById('btn-close-error');
  const btnSwitchToFlash = document.getElementById('btn-switch-to-flash');
  const btnSwitchToFlashLite = document.getElementById('btn-switch-to-flash-lite');
  const btnViewRawJson = document.getElementById('btn-view-raw-json');

  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');
  
  const loadingState = document.getElementById('loading-state');
  const loadingText = document.getElementById('loading-text');
  const emptyState = document.getElementById('empty-state');
  
  const outputEditableText = document.getElementById('output-editable-text');
  const outputCharCounter = document.getElementById('output-char-counter');
  const liveTagPreview = document.getElementById('live-tag-preview');

  const segmentedEditableText = document.getElementById('segmented-editable-text');
  const segmentedCharCounter = document.getElementById('segmented-char-counter');
  const btnCopySegmented = document.getElementById('btn-copy-segmented');
  const chkUseSegmented = document.getElementById('chk-use-segmented');
  const segFormatButtons = document.querySelectorAll('.seg-format-btn');
  
  const geminiPromptSection = document.getElementById('gemini-prompt-section');
  const geminiPromptEditable = document.getElementById('gemini-prompt-editable');
  
  const segmentsContainer = document.getElementById('segments-container');
  const rawJson = document.getElementById('raw-json');
  const btnCopyJson = document.getElementById('btn-copy-json');
  const jsonStatusBadge = document.getElementById('json-status-badge');

  const btnCopyOutput = document.getElementById('btn-copy-output');
  const btnCopyPrompt = document.getElementById('btn-copy-prompt');
  const tagInsertButtons = document.querySelectorAll('.tag-insert-btn');

  const audioPlayerCard = document.getElementById('audio-player-card');
  const multiPlayerContainer = document.getElementById('multi-player-container');
  
  const playerBoxLoraOn = document.getElementById('player-box-lora-on');
  const audioPlayerLoraOn = document.getElementById('audio-player-lora-on');
  const btnDownloadLoraOn = document.getElementById('btn-download-lora-on');

  const playerBoxLoraOff = document.getElementById('player-box-lora-off');
  const audioPlayerLoraOff = document.getElementById('audio-player-lora-off');
  const btnDownloadLoraOff = document.getElementById('btn-download-lora-off');

  const playerBoxNoEmotion = document.getElementById('player-box-no-emotion');
  const audioPlayerNoEmotion = document.getElementById('audio-player-no-emotion');
  const btnDownloadNoEmotion = document.getElementById('btn-download-no-emotion');

  const playerBoxRaw = document.getElementById('player-box-raw');
  const audioPlayerRaw = document.getElementById('audio-player-raw');
  const btnDownloadRaw = document.getElementById('btn-download-raw');

  const loraToggleOptions = document.querySelectorAll('.lora-toggle-opt');
  const loraCheckInputs = document.querySelectorAll('input[name="lora_mode_check"]');

  // Post-Process DSP module
  const chkPostProcess = document.getElementById('chk-post-process');
  const dspSection = document.getElementById('dsp-section');
  const dspParams = document.getElementById('dsp-params');
  const dspOffNote = document.getElementById('dsp-off-note');
  const dspStateSub = document.getElementById('dsp-state-sub');
  const btnDspReset = document.getElementById('btn-dsp-reset');
  const dspPresetButtons = document.querySelectorAll('.dsp-pill-btn');
  // Every control that maps onto a PostProcessParams field carries data-dsp-key,
  // so collecting and resetting stay table-driven instead of listing ids twice.
  const dspControls = document.querySelectorAll('[data-dsp-key]');
  const dspToneEnergy = document.querySelectorAll('.dsp-tone-energy');
  const dspToneRate = document.querySelectorAll('.dsp-tone-rate');
  const dspMatchEnergy = document.getElementById('dsp-match-energy');
  const dspMatchRate = document.getElementById('dsp-match-rate');

  let selectedAudioFile = null;
  let currentAudioUrlOn = null;
  let currentAudioUrlOff = null;
  let currentAudioUrlNoEmotion = null;
  let currentAudioUrlRaw = null;
  // Last /speak payload, kept so the full/short toggle can re-render without refetching.
  let lastRenderData = null;
  let segFormat = 'full';

  function getTimestamp() {
    const now = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    return `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
  }

  // --- Post-Process DSP -----------------------------------------------------

  // A control's on-screen unit is not always the API's unit (ms vs s, % vs ratio);
  // data-dsp-scale carries the multiplier so the label can stay human-readable.
  function dspValue(el) {
    if (el.type === 'checkbox') return el.checked;
    const scale = parseFloat(el.dataset.dspScale || '1');
    const raw = parseFloat(el.value);
    return Number.isFinite(raw) ? raw * scale : null;
  }

  function isDspEnabled() {
    return !chkPostProcess || chkPostProcess.checked;
  }

  // Only send what the user actually moved. Anything left at its default stays out
  // of the payload so the measured constants in audio_post.py remain authoritative.
  function collectDspParams() {
    if (!isDspEnabled()) return null;
    const params = {};

    dspControls.forEach((el) => {
      const key = el.dataset.dspKey;
      if (!key) return;
      const def = el.dataset.dspDefault;
      if (el.type === 'checkbox') {
        if (String(el.checked) !== def) params[key] = el.checked;
        return;
      }
      if (parseFloat(el.value) !== parseFloat(def)) {
        const v = dspValue(el);
        if (v !== null) params[key] = v;
      }
    });

    const energy = {};
    dspToneEnergy.forEach((el) => {
      const v = parseFloat(el.value);
      if (Number.isFinite(v) && v !== parseFloat(el.dataset.dspDefault)) {
        energy[el.dataset.tone] = v;
      }
    });
    if (Object.keys(energy).length) params.tone_energy_db = energy;

    const rate = {};
    dspToneRate.forEach((el) => {
      const v = parseFloat(el.value);
      if (Number.isFinite(v) && v !== parseFloat(el.dataset.dspDefault)) {
        rate[el.dataset.tone] = v;
      }
    });
    if (Object.keys(rate).length) params.tone_duration_ratio = rate;

    return Object.keys(params).length ? params : null;
  }

  // Sliders own their own readout span, named val-<slider id>.
  function syncDspLabel(el) {
    const out = document.getElementById(`val-${el.id}`);
    if (!out) return;
    const step = parseFloat(el.step || '1');
    const v = parseFloat(el.value);
    out.textContent = step < 0.05 ? v.toFixed(2) : (step < 1 ? v.toFixed(2) : String(v));
  }

  function syncAllDspLabels() {
    dspControls.forEach((el) => {
      if (el.type === 'range') syncDspLabel(el);
    });
  }

  // Each column of the per-tone table is only read when its matcher is on, so a
  // value typed into a dead column would silently do nothing. Disable it instead.
  function syncToneTableEnabled() {
    const energyOn = !dspMatchEnergy || dspMatchEnergy.checked;
    const rateOn = !dspMatchRate || dspMatchRate.checked;
    dspToneEnergy.forEach((el) => {
      el.disabled = !energyOn;
      el.title = energyOn ? '' : 'ปิด "เปิดปรับระดับเสียง" อยู่ — ค่าคอลัมน์นี้ไม่ถูกใช้';
    });
    dspToneRate.forEach((el) => {
      el.disabled = !rateOn;
      el.title = rateOn ? '' : 'ปิด "เปิดปรับความเร็ว" อยู่ — ค่าคอลัมน์นี้ไม่ถูกใช้';
    });
    const details = document.getElementById('dsp-tone-details');
    if (details) details.classList.toggle('dsp-tone-dead', !energyOn && !rateOn);
    const energySlider = document.getElementById('dsp-energy-match');
    if (energySlider) energySlider.disabled = !energyOn;
    const stretchSlider = document.getElementById('dsp-max-stretch');
    if (stretchSlider) stretchSlider.disabled = !rateOn;
  }

  function updateDspVisibility() {
    const on = isDspEnabled();
    if (dspParams) dspParams.classList.toggle('hidden', !on);
    if (dspOffNote) dspOffNote.classList.toggle('hidden', on);
    if (dspSection) dspSection.classList.toggle('dsp-disabled', !on);
    if (dspStateSub) {
      dspStateSub.textContent = on
        ? 'เปิดอยู่ — ปรับระดับเสียง จังหวะ และช่องว่างตามอารมณ์'
        : 'ปิดอยู่ — ต่อเสียงดิบจาก TTS ตรงๆ';
    }

    // With the module off, every mode already renders raw, so the raw variant is
    // a duplicate take -- and each take is a full generation. Lock it out rather
    // than let it spend the GPU twice for the same thing.
    const rawInput = document.querySelector('input[value="raw_tts"]');
    const rawOpt = document.getElementById('opt-raw-tts');
    if (rawInput) {
      rawInput.disabled = !on;
      if (!on && rawInput.checked) {
        rawInput.checked = false;
        if (rawOpt) rawOpt.classList.remove('active');
      }
    }
    if (rawOpt) {
      rawOpt.classList.toggle('opt-locked', !on);
      rawOpt.title = on ? '' : 'ปิด DSP อยู่ — ทุกโหมดเป็นเสียงดิบหมดแล้ว ไม่ต้องสร้างซ้ำ';
    }
  }

  const DSP_PRESETS = {
    reference: {},
    narration: {
      'dsp-gap-same': 0.14, 'dsp-gap-emotion': 0.30, 'dsp-gap-para': 0.70,
      'dsp-energy-match': 0.85, 'dsp-max-stretch': 12
    },
    dramatic: {
      'dsp-gap-same': 0.30, 'dsp-gap-emotion': 0.70, 'dsp-gap-para': 1.60,
      'dsp-energy-match': 0.55, 'dsp-max-stretch': 20
    },
    minimal: {
      'dsp-gap-same': 0.20, 'dsp-gap-emotion': 0.45, 'dsp-gap-para': 1.20,
      'dsp-energy-match': 0.20, 'dsp-max-stretch': 5, 'dsp-match-rate': false
    },
    // The control take for "is the emotion layer earning its place". Keeps every
    // cleanup step -- DC removal, trimming, edge fades, the gap policy, peak
    // limiting -- and drops all three things that shape emotion: the per-tone dB
    // table, the per-tone pace table, and the extra pause at an emotion change.
    cleanup: {
      'dsp-match-energy': false,
      'dsp-match-rate': false,
      'dsp-gap-emotion': 0.20
    }
  };

  function resetDspControls() {
    dspControls.forEach((el) => {
      const def = el.dataset.dspDefault;
      if (def === undefined) return;
      if (el.type === 'checkbox') el.checked = def === 'true';
      else el.value = def;
    });
    [...dspToneEnergy, ...dspToneRate].forEach((el) => { el.value = el.dataset.dspDefault; });
    syncAllDspLabels();
  }

  function applyDspPreset(name) {
    resetDspControls();
    const preset = DSP_PRESETS[name] || {};
    Object.entries(preset).forEach(([id, value]) => {
      const el = document.getElementById(id);
      if (!el) return;
      if (el.type === 'checkbox') el.checked = !!value;
      else el.value = value;
    });
    syncAllDspLabels();
    syncToneTableEnabled();
  }

  if (chkPostProcess) {
    chkPostProcess.addEventListener('change', updateDspVisibility);
  }
  dspControls.forEach((el) => {
    if (el.type === 'range') {
      el.addEventListener('input', () => {
        syncDspLabel(el);
        dspPresetButtons.forEach(b => b.classList.remove('active'));
      });
    }
  });
  [dspMatchEnergy, dspMatchRate].forEach((el) => {
    if (!el) return;
    el.addEventListener('change', () => {
      syncToneTableEnabled();
      dspPresetButtons.forEach(b => b.classList.remove('active'));
    });
  });
  dspPresetButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      applyDspPreset(btn.dataset.dspPreset);
      dspPresetButtons.forEach(b => b.classList.toggle('active', b === btn));
    });
  });
  if (btnDspReset) {
    btnDspReset.addEventListener('click', () => {
      resetDspControls();
      syncToneTableEnabled();
      dspPresetButtons.forEach(b => b.classList.toggle('active', b.dataset.dspPreset === 'reference'));
      if (chkPostProcess) chkPostProcess.checked = true;
      updateDspVisibility();
    });
  }
  syncAllDspLabels();
  syncToneTableEnabled();
  updateDspVisibility();

  function getSelectedGenModes() {
    const checked = [...document.querySelectorAll('input[name="lora_mode_check"]:checked')].map(el => el.value);
    return checked.length > 0 ? checked : ['lora_on'];
  }

  loraCheckInputs.forEach(input => {
    input.addEventListener('change', () => {
      const checkedBoxes = document.querySelectorAll('input[name="lora_mode_check"]:checked');
      if (checkedBoxes.length === 0) {
        input.checked = true;
      }
      loraCheckInputs.forEach(inp => {
        const parent = inp.closest('.lora-toggle-opt');
        if (parent) {
          parent.classList.toggle('active', inp.checked);
        }
      });
    });
  });

  // Presets Data
  const PRESETS = {
    calm: {
      text: '[calm] หายใจเข้าลึกๆ ผ่อนคลาย แล้วค่อยๆ ปล่อยวางทุกอย่างลงนะ',
      guidance: 'สงบ นุ่มนวล ช้าๆ ผ่อนคลาย'
    },
    shift: {
      text: 'ขอโทษนะ ฉันไม่ได้ตั้งใจ แต่เธอก็ไม่ฟังฉันเลย',
      guidance: 'ท่อนแรกขอเศร้าขอโทษจากใจ ท่อนหลังตัดพ้อโกรธ'
    },
    sarcastic: {
      text: 'แหม เก่งจังเลยนะ ทำพังหมดทั้งห้องแล้วเนี่ย',
      guidance: 'ประชดประชันแดกดันอย่างแรง'
    },
    happy: {
      text: 'ยินดีด้วยนะ! ในที่สุดก็ทำสำเร็จแล้ว สุดยอดไปเลย!',
      guidance: 'ดีใจสุดขีด ร่าเริงมาก'
    },
    news: {
      text: 'กรมอุตุนิยมวิทยาประกาศเตือน จะมีฝนตกหนักถึงหนักมากในหลายพื้นที่ ประชาชนควรระมัดระวังน้ำท่วมฉับพลัน',
      guidance: 'อ่านข่าว สุภาพ เป็นทางการ เป็นกลาง'
    }
  };

  // API Base URL
  // Served over http the API is same-origin, so the base stays empty. The literal
  // only applies when index.html is opened straight off disk, and it has to name
  // the studio's own port (8011) -- 8000 was the retired model service.
  const API_BASE = window.location.protocol === 'file:' ? 'http://127.0.0.1:8011' : '';

  // Get selected model helper
  function getSelectedModel() {
    if (!llmModelSelect) return null;
    const val = llmModelSelect.value;
    if (val === 'custom') {
      return (customModelInput && customModelInput.value.trim()) ? customModelInput.value.trim() : null;
    }
    return val || null;
  }

  const btnRefreshModels = document.getElementById('btn-refresh-models');

  async function loadAvailableModels(forceRefresh = false) {
    if (!llmModelSelect) return;
    if (btnRefreshModels) btnRefreshModels.classList.add('spinning');

    try {
      const url = `${API_BASE}/models${forceRefresh ? '?refresh=true' : ''}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`Failed to load models: ${res.status}`);
      const data = await res.json();

      const currentSelected = llmModelSelect.value;
      llmModelSelect.innerHTML = '';

      // 1. 9arm Gateway / OpenAI-Compatible models
      const openaiData = data.providers?.openai;
      if (openaiData && openaiData.available && openaiData.models && openaiData.models.length > 0) {
        const group = document.createElement('optgroup');
        group.label = `9arm Gateway / OpenAI Compatible (API Key Active)`;
        openaiData.models.forEach(m => {
          const opt = document.createElement('option');
          opt.value = m.id;
          opt.textContent = m.name;
          group.appendChild(opt);
        });
        llmModelSelect.appendChild(group);
      }

      // 2. Google Gemini models
      const geminiData = data.providers?.gemini;
      if (geminiData && geminiData.models && geminiData.models.length > 0) {
        const group = document.createElement('optgroup');
        group.label = `Google Gemini (${geminiData.available ? 'API Key Active' : 'Default'})`;
        geminiData.models.forEach(m => {
          const opt = document.createElement('option');
          opt.value = m.id;
          opt.textContent = m.name;
          group.appendChild(opt);
        });
        llmModelSelect.appendChild(group);
      }

      // 3. Anthropic Claude models (if available or configured)
      const anthropicData = data.providers?.anthropic;
      if (anthropicData && anthropicData.available && anthropicData.models && anthropicData.models.length > 0) {
        const group = document.createElement('optgroup');
        group.label = `Anthropic Claude (API Key Active)`;
        anthropicData.models.forEach(m => {
          const opt = document.createElement('option');
          opt.value = m.id;
          opt.textContent = m.name;
          group.appendChild(opt);
        });
        llmModelSelect.appendChild(group);
      }

      // 4. Custom Option
      const customOpt = document.createElement('option');
      customOpt.value = 'custom';
      customOpt.textContent = '✏️ ระบุชื่อโมเดลเอง (Custom)...';
      llmModelSelect.appendChild(customOpt);

      // Restore selection or set default
      const defaultChoice = data.default_model || 'gemini-3.6-flash';
      if (currentSelected && [...llmModelSelect.options].some(o => o.value === currentSelected)) {
        llmModelSelect.value = currentSelected;
      } else if ([...llmModelSelect.options].some(o => o.value === defaultChoice)) {
        llmModelSelect.value = defaultChoice;
      }

      if (forceRefresh && btnRefreshModels) {
        const spanEl = btnRefreshModels.querySelector('span');
        if (spanEl) {
          spanEl.textContent = '✅ อัปเดตแล้ว!';
          setTimeout(() => {
            spanEl.textContent = '🔄 รีเฟรชโมเดล';
          }, 1800);
        }
      }
    } catch (e) {
      console.warn('Could not load models list:', e);
    } finally {
      if (btnRefreshModels) btnRefreshModels.classList.remove('spinning');
    }
  }

  if (btnRefreshModels) {
    btnRefreshModels.addEventListener('click', () => {
      loadAvailableModels(true);
    });
  }

  // Model select change handler
  if (llmModelSelect) {
    llmModelSelect.addEventListener('change', () => {
      if (llmModelSelect.value === 'custom') {
        customModelGroup.classList.remove('hidden');
        customModelInput.focus();
      } else {
        customModelGroup.classList.add('hidden');
      }
    });
  }

  // CFG & Parameter Slider & Stepper updates
  function updateCfgValue(newVal) {
    const clamped = Math.max(1.0, Math.min(6.0, parseFloat(newVal) || 2.5));
    const rounded = Math.round(clamped * 10) / 10;
    if (paramCfg) paramCfg.value = rounded;
    if (valCfg) valCfg.textContent = rounded.toFixed(1);

    // Highlight matching pill
    cfgPillButtons.forEach(btn => {
      const pillVal = parseFloat(btn.getAttribute('data-cfg'));
      btn.classList.toggle('active', Math.abs(pillVal - rounded) < 0.05);
    });
  }

  if (paramCfg) {
    paramCfg.addEventListener('input', () => {
      updateCfgValue(paramCfg.value);
    });
  }

  if (btnCfgMinus) {
    btnCfgMinus.addEventListener('click', () => {
      const current = parseFloat(paramCfg.value) || 2.5;
      updateCfgValue(current - 0.1);
    });
  }

  if (btnCfgPlus) {
    btnCfgPlus.addEventListener('click', () => {
      const current = parseFloat(paramCfg.value) || 2.5;
      updateCfgValue(current + 0.1);
    });
  }

  cfgPillButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetVal = parseFloat(btn.getAttribute('data-cfg'));
      if (!isNaN(targetVal)) {
        updateCfgValue(targetVal);
      }
    });
  });

  if (paramSteps && valSteps) {
    paramSteps.addEventListener('input', () => {
      valSteps.textContent = paramSteps.value;
    });
  }

  // Check API Health & Fetch Speakers
  async function checkHealthAndSpeakers() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        const data = await res.json();
        healthStatus.className = 'status-indicator online';
        healthStatus.querySelector('.status-label').textContent = `API พร้อมใช้งาน (${data.speakers_count || 0} voices, LLM: ${data.default_model || data.provider})`;
      } else {
        throw new Error('Health check failed');
      }
    } catch (e) {
      healthStatus.className = 'status-indicator offline';
      healthStatus.querySelector('.status-label').textContent = 'ไม่สามารถเชื่อมต่อ API ได้ (โปรดตรวจสอบสถานะเซิร์ฟเวอร์)';
    }

    loadSpeakersList();
    loadAvailableModels(false);
  }

  async function loadSpeakersList() {
    try {
      const res = await fetch(`${API_BASE}/speakers`);
      if (res.ok) {
        const data = await res.json();
        const prevVal = speakerSelect.value;
        speakerSelect.innerHTML = '<option value="">-- ไม่ใช้เสียงโคลน (Base Voice) --</option>';
        (data.speakers || []).forEach(spk => {
          const opt = document.createElement('option');
          opt.value = spk.id;
          opt.textContent = `🎙️ ${spk.name} (${spk.filename})`;
          speakerSelect.appendChild(opt);
        });
        if (prevVal) speakerSelect.value = prevVal;
        syncSpeakerControls();
      }
    } catch (e) {
      console.warn('Could not load speakers:', e);
    }
  }

  function setButtonPlayingState(btn, isPlaying, defaultText = 'ฟังเสียง Ref', playingText = 'หยุด') {
    if (!btn) return;
    btn.classList.toggle('playing', isPlaying);
    const playIcon = btn.querySelector('.play-icon');
    const pauseIcon = btn.querySelector('.pause-icon');
    const label = btn.querySelector('.ref-btn-label') || btn.querySelector('.upload-btn-label') || btn.querySelector('span:last-child');
    if (playIcon) playIcon.classList.toggle('hidden', isPlaying);
    if (pauseIcon) pauseIcon.classList.toggle('hidden', !isPlaying);
    if (label) label.textContent = isPlaying ? playingText : defaultText;
  }

  function stopAllAudios(exceptEl = null) {
    const audios = [
      speakerRefAudio,
      uploadedRefAudio,
      audioPlayerLoraOn,
      audioPlayerLoraOff,
      audioPlayerNoEmotion
    ];
    audios.forEach(a => {
      if (a && a !== exceptEl && !a.paused) {
        a.pause();
      }
    });
  }

  // Sync speaker select controls: Re-roll visibility and Ref audio player button
  function syncSpeakerControls() {
    const hasSpeaker = Boolean(speakerSelect.value);
    if (btnRerollSeed) btnRerollSeed.classList.toggle('hidden', hasSpeaker);
    if (btnPlaySpeakerRef) {
      btnPlaySpeakerRef.classList.toggle('hidden', !hasSpeaker);
      setButtonPlayingState(btnPlaySpeakerRef, false, 'ฟังเสียง Ref', 'หยุด');
    }
    if (speakerRefAudio) {
      speakerRefAudio.pause();
      if (hasSpeaker) {
        speakerRefAudio.src = `${API_BASE}/speakers/${encodeURIComponent(speakerSelect.value)}/audio`;
        speakerRefAudio.load();
      } else {
        speakerRefAudio.removeAttribute('src');
      }
    }
  }

  if (btnPlaySpeakerRef && speakerRefAudio) {
    btnPlaySpeakerRef.addEventListener('click', async (e) => {
      e.stopPropagation();
      const sid = speakerSelect.value;
      if (!sid) return;

      if (speakerRefAudio.paused) {
        stopAllAudios(speakerRefAudio);
        try {
          if (!speakerRefAudio.src || !speakerRefAudio.src.includes(encodeURIComponent(sid))) {
            speakerRefAudio.src = `${API_BASE}/speakers/${encodeURIComponent(sid)}/audio`;
          }
          await speakerRefAudio.play();
          setButtonPlayingState(btnPlaySpeakerRef, true, 'ฟังเสียง Ref', 'หยุด');
        } catch (err) {
          console.warn('Could not play speaker audio:', err);
          alert('ไม่สามารถเล่นเสียงตัวอย่างได้ (อาจยังไม่มีไฟล์เสียงสำหรับ Speaker นี้บนเซิร์ฟเวอร์)');
          setButtonPlayingState(btnPlaySpeakerRef, false, 'ฟังเสียง Ref', 'หยุด');
        }
      } else {
        speakerRefAudio.pause();
        speakerRefAudio.currentTime = 0;
        setButtonPlayingState(btnPlaySpeakerRef, false, 'ฟังเสียง Ref', 'หยุด');
      }
    });

    speakerRefAudio.addEventListener('ended', () => {
      setButtonPlayingState(btnPlaySpeakerRef, false, 'ฟังเสียง Ref', 'หยุด');
    });

    speakerRefAudio.addEventListener('pause', () => {
      setButtonPlayingState(btnPlaySpeakerRef, false, 'ฟังเสียง Ref', 'หยุด');
    });

    speakerRefAudio.addEventListener('play', () => {
      setButtonPlayingState(btnPlaySpeakerRef, true, 'ฟังเสียง Ref', 'หยุด');
    });
  }

  // Base Voice is one cached generation reused for every unpinned request, so a bad
  // draw survives re-rendering and restarts. This throws it away; the next synthesis
  // mints a new speaker. It does not touch cloned profiles in ref/.
  async function rerollSeedVoice() {
    if (!btnRerollSeed) return;
    const label = btnRerollSeed.textContent;
    btnRerollSeed.disabled = true;
    btnRerollSeed.textContent = 'กำลังสุ่ม...';
    try {
      const res = await fetch(`${API_BASE}/speakers/seed/reroll`, { method: 'POST' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      alert(data.cache_removed
        ? 'สุ่มเสียงใหม่แล้ว — กดสังเคราะห์อีกครั้งเพื่อฟังผู้พูดคนใหม่'
        : 'ยังไม่มีเสียงที่แคชไว้ — ครั้งถัดไปจะสุ่มผู้พูดใหม่อยู่แล้ว');
    } catch (e) {
      alert(`สุ่มเสียงใหม่ไม่สำเร็จ: ${e.message}`);
    } finally {
      btnRerollSeed.disabled = false;
      btnRerollSeed.textContent = label;
    }
  }

  if (btnRerollSeed) btnRerollSeed.addEventListener('click', rerollSeedVoice);
  speakerSelect.addEventListener('change', syncSpeakerControls);
  syncSpeakerControls();

  checkHealthAndSpeakers();
  setInterval(checkHealthAndSpeakers, 30000);

  // Engine Switch Visibility
  function updateEngineVisibility() {
    const eng = engineSelect.value;
    if (eng === 'voxcpm' || eng === 'siangtts') {
      speakerGroup.classList.remove('hidden');
      uploadVoiceArea.classList.remove('hidden');
    } else {
      speakerGroup.classList.add('hidden');
      uploadVoiceArea.classList.add('hidden');
      if (speakerRefAudio) speakerRefAudio.pause();
      if (uploadedRefAudio) uploadedRefAudio.pause();
    }
  }
  engineSelect.addEventListener('change', updateEngineVisibility);
  updateEngineVisibility();

  // File Upload & Drag-and-Drop & Audio Preview
  let uploadedAudioBlobUrl = null;

  if (btnPlayUploadedRef && uploadedRefAudio) {
    btnPlayUploadedRef.addEventListener('click', async (e) => {
      e.stopPropagation();
      if (!uploadedRefAudio.src) return;

      if (uploadedRefAudio.paused) {
        stopAllAudios(uploadedRefAudio);
        try {
          await uploadedRefAudio.play();
          setButtonPlayingState(btnPlayUploadedRef, true, 'ฟังเสียง Ref', 'หยุด');
        } catch (err) {
          console.warn('Could not play uploaded audio:', err);
          setButtonPlayingState(btnPlayUploadedRef, false, 'ฟังเสียง Ref', 'หยุด');
        }
      } else {
        uploadedRefAudio.pause();
        uploadedRefAudio.currentTime = 0;
        setButtonPlayingState(btnPlayUploadedRef, false, 'ฟังเสียง Ref', 'หยุด');
      }
    });

    uploadedRefAudio.addEventListener('ended', () => {
      setButtonPlayingState(btnPlayUploadedRef, false, 'ฟังเสียง Ref', 'หยุด');
    });

    uploadedRefAudio.addEventListener('pause', () => {
      setButtonPlayingState(btnPlayUploadedRef, false, 'ฟังเสียง Ref', 'หยุด');
    });

    uploadedRefAudio.addEventListener('play', () => {
      setButtonPlayingState(btnPlayUploadedRef, true, 'ฟังเสียง Ref', 'หยุด');
    });
  }

  dropZone.addEventListener('click', () => {
    audioFileInput.click();
  });

  audioFileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleAudioFileSelected(e.target.files[0]);
    }
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('drag-over');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleAudioFileSelected(e.dataTransfer.files[0]);
    }
  });

  function handleAudioFileSelected(file) {
    selectedAudioFile = file;
    if (uploadedAudioBlobUrl) {
      URL.revokeObjectURL(uploadedAudioBlobUrl);
    }
    uploadedAudioBlobUrl = URL.createObjectURL(file);
    if (uploadedRefAudio) {
      uploadedRefAudio.src = uploadedAudioBlobUrl;
      uploadedRefAudio.load();
    }
    if (btnPlayUploadedRef) {
      setButtonPlayingState(btnPlayUploadedRef, false, 'ฟังเสียง Ref', 'หยุด');
    }
    selectedFilename.textContent = `🎵 ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    dropZoneContent.classList.add('hidden');
    fileInfoBadge.classList.remove('hidden');
    btnSaveSpeaker.classList.remove('hidden');
  }

  btnRemoveFile.addEventListener('click', (e) => {
    e.stopPropagation();
    selectedAudioFile = null;
    if (uploadedRefAudio) {
      uploadedRefAudio.pause();
      uploadedRefAudio.removeAttribute('src');
    }
    if (uploadedAudioBlobUrl) {
      URL.revokeObjectURL(uploadedAudioBlobUrl);
      uploadedAudioBlobUrl = null;
    }
    if (btnPlayUploadedRef) {
      setButtonPlayingState(btnPlayUploadedRef, false, 'ฟังเสียง Ref', 'หยุด');
    }
    audioFileInput.value = '';
    dropZoneContent.classList.remove('hidden');
    fileInfoBadge.classList.add('hidden');
    btnSaveSpeaker.classList.add('hidden');
  });

  // Save Speaker Profile
  btnSaveSpeaker.addEventListener('click', async (e) => {
    e.stopPropagation();
    if (!selectedAudioFile) return;

    const defaultName = selectedAudioFile.name.replace(/\.[^/.]+$/, "");
    const speakerName = prompt("ตั้งชื่อโปรไฟล์เสียง (Speaker ID / Name):", defaultName);
    if (!speakerName) return;

    const formData = new FormData();
    formData.append('file', selectedAudioFile);
    formData.append('speaker_id', speakerName);

    try {
      btnSaveSpeaker.disabled = true;
      btnSaveSpeaker.textContent = "กำลังบันทึก...";
      const res = await fetch(`${API_BASE}/speakers`, {
        method: 'POST',
        body: formData
      });
      if (!res.ok) throw new Error("Upload failed");
      const result = await res.json();
      alert(`บันทึกเสียง '${result.name}' สำเร็จ!`);
      await loadSpeakersList();
      speakerSelect.value = result.id;
    } catch (err) {
      alert(`ไม่สามารถบันทึกเสียงได้: ${err.message}`);
    } finally {
      btnSaveSpeaker.disabled = false;
      btnSaveSpeaker.textContent = "บันทึกเป็นโปรไฟล์เสียง";
    }
  });

  // Character Counter for Input
  textInput.addEventListener('input', () => {
    const len = textInput.value.length;
    charCounter.textContent = `${len.toLocaleString()} ตัวอักษร`;
  });

  // Character Counter & Live Highlight for Output Textarea
  function updateOutputPreview() {
    const val = outputEditableText.value;
    outputCharCounter.textContent = `${val.length.toLocaleString()} ตัวอักษร`;
    liveTagPreview.innerHTML = highlightAudioTags(escapeHtml(val)) || '<span style="color:var(--text-muted);">ไม่มีข้อความ</span>';
  }

  outputEditableText.addEventListener('input', updateOutputPreview);

  // Group consecutive segments sharing a tone, mirroring how the VoxCPM renderer
  // builds chunks -- so run i lines up with data.chunks[i].
  function buildToneRuns(segments) {
    const runs = [];
    (segments || []).forEach(seg => {
      const last = runs[runs.length - 1];
      if (last && last.tone === seg.tone) {
        last.text += seg.text;
      } else {
        runs.push({ tone: seg.tone, intensity: seg.intensity, text: seg.text });
      }
    });
    return runs.map(r => ({ ...r, text: r.text.trim() })).filter(r => r.text);
  }

  function buildSegmentedText(data, format) {
    if (!data) return '';
    const runs = buildToneRuns(data.segments);
    const chunks = data.chunks || [];
    return runs.map((run, i) => {
      const body = (chunks[i] && chunks[i].body) ? chunks[i].body : run.text;
      if (format === 'short') {
        if (run.tone === 'neutral') return body;
        return `(${run.tone.charAt(0).toUpperCase()}${run.tone.slice(1)})${body}`;
      }
      const instruction = chunks[i] ? chunks[i].instruction : null;
      return instruction ? `${instruction}${body}` : body;
    }).join('');
  }

  function updateSegmentedPreview() {
    if (!segmentedEditableText) return;
    segmentedCharCounter.textContent = `${segmentedEditableText.value.length.toLocaleString()} ตัวอักษร`;
  }

  function renderSegmentedEditor() {
    if (!segmentedEditableText) return;
    segmentedEditableText.value = buildSegmentedText(lastRenderData, segFormat);
    updateSegmentedPreview();
  }

  if (segmentedEditableText) {
    segmentedEditableText.addEventListener('input', updateSegmentedPreview);
  }

  segFormatButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      segFormat = btn.getAttribute('data-format');
      segFormatButtons.forEach(b => b.classList.toggle('active', b === btn));
      renderSegmentedEditor();
    });
  });

  // Presets click
  presetButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.getAttribute('data-preset');
      if (PRESETS[key]) {
        textInput.value = PRESETS[key].text;
        guidanceInput.value = PRESETS[key].guidance || '';
        textInput.dispatchEvent(new Event('input'));
        textInput.focus();
      }
    });
  });

  // Clear button
  btnClear.addEventListener('click', () => {
    textInput.value = '';
    guidanceInput.value = '';
    updateCfgValue(2.5);
    textInput.dispatchEvent(new Event('input'));
    showEmptyState();
  });

  // Tab switching
  function switchTab(targetTab) {
    tabButtons.forEach(b => {
      if (b.getAttribute('data-tab') === targetTab) {
        b.classList.add('active');
      } else {
        b.classList.remove('active');
      }
    });

    tabPanes.forEach(pane => {
      if (pane.id === `tab-${targetTab}`) {
        pane.classList.remove('hidden');
      } else {
        pane.classList.add('hidden');
      }
    });
  }

  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');
      switchTab(targetTab);
    });
  });

  function showLoading(isLoading, customText = 'กำลังวิเคราะห์ข้อความด้วย LLM...') {
    if (isLoading) {
      loadingText.textContent = customText;
      loadingState.classList.remove('hidden');
      emptyState.classList.add('hidden');
      tabPanes.forEach(pane => pane.classList.add('hidden'));
      btnProcess.disabled = true;
      btnSynthesizeDirect.disabled = true;
    } else {
      loadingState.classList.add('hidden');
      btnProcess.disabled = false;
      btnSynthesizeDirect.disabled = false;
    }
  }

  function showEmptyState() {
    emptyState.classList.remove('hidden');
    loadingState.classList.add('hidden');
    tabPanes.forEach(pane => pane.classList.add('hidden'));
    modelBadge.classList.add('hidden');
    outputEditableText.value = '';
    liveTagPreview.innerHTML = '';
    lastRenderData = null;
    if (segmentedEditableText) {
      segmentedEditableText.value = '';
      updateSegmentedPreview();
    }
    audioPlayerCard.classList.add('hidden');
    if (playerBoxLoraOn) playerBoxLoraOn.classList.add('hidden');
    if (playerBoxLoraOff) playerBoxLoraOff.classList.add('hidden');
    if (playerBoxNoEmotion) playerBoxNoEmotion.classList.add('hidden');
    if (playerBoxRaw) playerBoxRaw.classList.add('hidden');
    if (audioPlayerLoraOn) { audioPlayerLoraOn.pause(); audioPlayerLoraOn.currentTime = 0; }
    if (audioPlayerLoraOff) { audioPlayerLoraOff.pause(); audioPlayerLoraOff.currentTime = 0; }
    if (audioPlayerNoEmotion) { audioPlayerNoEmotion.pause(); audioPlayerNoEmotion.currentTime = 0; }
    if (audioPlayerRaw) { audioPlayerRaw.pause(); audioPlayerRaw.currentTime = 0; }
    if (speakerRefAudio) { speakerRefAudio.pause(); speakerRefAudio.currentTime = 0; }
    if (uploadedRefAudio) { uploadedRefAudio.pause(); uploadedRefAudio.currentTime = 0; }
    if (btnPlaySpeakerRef) { setButtonPlayingState(btnPlaySpeakerRef, false, 'ฟังเสียง Ref', 'หยุด'); }
    if (btnPlayUploadedRef) { setButtonPlayingState(btnPlayUploadedRef, false, 'ฟังเสียง Ref', 'หยุด'); }
    if (errorBanner) errorBanner.classList.add('hidden');
  }

  function formatIntensityStars(intensity) {
    if (intensity === 1) return '●○○ (Mild)';
    if (intensity === 3) return '●●● (Strong)';
    return '●●○ (Standard)';
  }

  // Mirrors STYLE_TAG_RE in app/renderers/voxcpm.py: same label vocabulary and the
  // same optional :1-3 intensity, so a tag the renderer will act on is a tag the
  // preview shows. The old pattern accepted letters and spaces only, which left
  // every intensity-carrying tag -- [sarcastic:3], [angry:3] -- as unmarked text.
  const TAG_RE = /\[\s*([A-Za-z\u0e00-\u0e7f][A-Za-z\u0e00-\u0e7f\s,.\-]*?)\s*(?::\s*([123]))?\s*\]/g;

  // The tag as one word, which is the point of the preview: the operator wants to
  // see what a tag condenses to at a glance, not re-read what they typed. Matches
  // the 'short' segmented format (capitalised, parenthesised) so the two agree.
  // The raw tag stays on the tooltip, since intensity still matters when testing.
  function shortTagLabel(label) {
    const word = label.trim().split(/\s+/)[0];
    return word.charAt(0).toUpperCase() + word.slice(1);
  }

  function highlightAudioTags(text) {
    // Instructions first: the tag pass emits parenthesised text of its own, and
    // running this after it would highlight that as a hand-written instruction.
    let formatted = text.replace(/(\([a-zA-Z\u0e00-\u0e7f\s,.-]+\))/g, '<span class="instruction-highlight">$1</span>');
    formatted = formatted.replace(TAG_RE, (raw, label, intensity) => {
      const short = shortTagLabel(label);
      const title = intensity ? `${raw.trim()} -> (${short}) intensity ${intensity}` : `${raw.trim()} -> (${short})`;
      return `<span class="tag-highlight" title="${title}">(${short})</span>`;
    });
    return formatted;
  }

  function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  // Tag Inserter Toolbar
  tagInsertButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const tag = btn.getAttribute('data-tag');
      insertTextAtCursor(outputEditableText, tag);
      updateOutputPreview();
      outputEditableText.focus();
    });
  });

  function insertTextAtCursor(textarea, textToInsert) {
    const startPos = textarea.selectionStart;
    const endPos = textarea.selectionEnd;
    const currentVal = textarea.value;

    textarea.value = currentVal.substring(0, startPos) + textToInsert + currentVal.substring(endPos);
    textarea.selectionStart = textarea.selectionEnd = startPos + textToInsert.length;
    textarea.dispatchEvent(new Event('input'));
  }

  // Error Banner Controls
  if (btnCloseError) {
    btnCloseError.addEventListener('click', () => {
      errorBanner.classList.add('hidden');
    });
  }

  if (btnSwitchToFlash) {
    btnSwitchToFlash.addEventListener('click', () => {
      if (llmModelSelect) {
        llmModelSelect.value = 'gemini-3.6-flash';
        customModelGroup.classList.add('hidden');
      }
      errorBanner.classList.add('hidden');
      handleAnnotate();
    });
  }

  if (btnSwitchToFlashLite) {
    btnSwitchToFlashLite.addEventListener('click', () => {
      if (llmModelSelect) {
        llmModelSelect.value = 'gemini-3.5-flash-lite';
        customModelGroup.classList.add('hidden');
      }
      errorBanner.classList.add('hidden');
      handleAnnotate();
    });
  }

  if (btnViewRawJson) {
    btnViewRawJson.addEventListener('click', () => {
      switchTab('raw');
    });
  }

  // Process Annotation & Render Script
  async function handleAnnotate() {
    const text = textInput.value.trim();
    if (!text) {
      alert('กรุณากรอกข้อความภาษาไทย หรือคลิกเลือกตัวอย่าง Preset ด้านบนก่อนกดวิเคราะห์');
      textInput.focus();
      return;
    }

    const guidance = guidanceInput.value.trim();
    const engine = engineSelect.value;
    const model = getSelectedModel();
    showLoading(true, `กำลังวิเคราะห์ข้อความด้วย ${model || 'LLM'}...`);

    try {
      const response = await fetch(`${API_BASE}/speak`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text: text,
          guidance: guidance || null,
          engine: engine,
          model: model
        })
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(data.detail || `Server error: ${response.status}`);
      }

      renderResults(data, engine);
    } catch (err) {
      if (errorBanner) {
        errorBanner.classList.remove('hidden');
        errorTitle.textContent = 'เกิดข้อผิดพลาดในการเรียก LLM API';
        errorDetailText.textContent = err.message || 'ไม่สามารถเชื่อมต่อเซิร์ฟเวอร์ได้';
      }
      if (rawJson) {
        rawJson.textContent = JSON.stringify({ error: err.message, status: "request_failed" }, null, 2);
      }
      if (jsonStatusBadge) {
        jsonStatusBadge.textContent = 'Error / Failed';
        jsonStatusBadge.className = 'json-status-badge error-badge';
      }
      alert(`เกิดข้อผิดพลาดในการประมวลผล: ${err.message}\n(ดูรายละเอียดเพิ่มเติมในแถบ Raw JSON)`);
      showEmptyState();
    } finally {
      showLoading(false);
    }
  }

  // Helper to strip style/emotion tags
  function stripEmotionTags(str) {
    if (!str) return '';
    return str
      .replace(/\[\s*[A-Za-z\u0e00-\u0e7f][A-Za-z\u0e00-\u0e7f\s,.\-]*?(?::\s*[123])?\s*\]/g, '')
      .replace(/\(\s*[A-Za-z\u0e00-\u0e7f][A-Za-z\u0e00-\u0e7f\s,.\-]*?(?::\s*[123])?\s*\)/g, '')
      .replace(/\s+/g, ' ')
      .trim();
  }

  // Synthesis Helper
  async function fetchSynthesisBlob({ text, speakerId, guidance, engine, model, cfgValue, timesteps, loraMode, autoAnnotate = true, postProcess = true, dspParams = null }) {
    if (selectedAudioFile) {
      const formData = new FormData();
      formData.append('text', text);
      formData.append('file', selectedAudioFile);
      if (guidance) formData.append('guidance', guidance);
      if (model) formData.append('model', model);
      formData.append('cfg_value', cfgValue);
      formData.append('inference_timesteps', timesteps);
      formData.append('auto_annotate', autoAnnotate ? 'true' : 'false');
      formData.append('lora_mode', loraMode);
      formData.append('post_process', postProcess ? 'true' : 'false');
      // Multipart has no nested objects, so the overrides ride along as JSON.
      if (postProcess && dspParams) {
        formData.append('post_process_params', JSON.stringify(dspParams));
      }

      const response = await fetch(`${API_BASE}/synthesize/upload`, {
        method: 'POST',
        body: formData
      });
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Synthesis failed: ${response.status}`);
      }
      return await response.blob();
    } else {
      const response = await fetch(`${API_BASE}/synthesize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text: text,
          speaker_id: speakerId,
          guidance: guidance || null,
          engine: engine,
          model: model,
          cfg_value: cfgValue,
          inference_timesteps: timesteps,
          auto_annotate: autoAnnotate,
          lora_mode: loraMode,
          post_process: postProcess,
          post_process_params: postProcess ? dspParams : null
        })
      });
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Synthesis failed: ${response.status}`);
      }
      return await response.blob();
    }
  }

  // Synthesize Speech
  async function handleSynthesize() {
    const segmentedText = segmentedEditableText ? segmentedEditableText.value.trim() : '';
    const useSegmented = chkUseSegmented && chkUseSegmented.checked && segmentedText;
    const text = useSegmented
      ? segmentedText
      : (outputEditableText.value.trim() || textInput.value.trim());
    if (!text) {
      alert('กรุณากรอกข้อความภาษาไทยก่อนสังเคราะห์เสียง');
      textInput.focus();
      return;
    }

    const engine = engineSelect.value;
    const speakerId = speakerSelect.value || null;
    const cfgValue = parseFloat(paramCfg.value) || 2.5;
    const timesteps = parseInt(paramSteps.value, 10) || 10;
    const guidance = guidanceInput.value.trim();
    const model = getSelectedModel();
    const selectedModes = getSelectedGenModes();
    const dspOn = isDspEnabled();
    const dspParams = collectDspParams();
    const ts = getTimestamp();
    const spkPrefix = speakerId ? `${speakerId}_` : '';

    const modeLabels = {
      lora_on: 'Thai LoRA (ON)',
      lora_off: 'LoRA OFF (Base)',
      no_emotion: 'ไม่ใส่อารมณ์ (เสียงเรียบ)',
      raw_tts: 'เสียงสด (Raw / No DSP)'
    };

    try {
      stopAllAudios();
      const modeNames = selectedModes.map(m => modeLabels[m] || m).join(', ');
      if (selectedModes.length > 1) {
        showLoading(true, `⚡ กำลังสร้าง ${selectedModes.length} รูปแบบพร้อมกัน (${modeNames})...`);
      } else {
        showLoading(true, `🎙️ กำลังสังเคราะห์เสียงด้วย SiangTTS [${modeNames}]...`);
      }

      // Execute synthesis for all selected modes concurrently
      const tasks = selectedModes.map(async (mode) => {
        let blob;
        if (mode === 'lora_on') {
          blob = await fetchSynthesisBlob({
            text, speakerId, guidance, engine, model, cfgValue, timesteps, loraMode: 'on', autoAnnotate: true, postProcess: dspOn, dspParams
          });
        } else if (mode === 'lora_off') {
          blob = await fetchSynthesisBlob({
            text, speakerId, guidance, engine, model, cfgValue, timesteps, loraMode: 'off', autoAnnotate: true, postProcess: dspOn, dspParams
          });
        } else if (mode === 'no_emotion') {
          const plainText = stripEmotionTags(text) || text;
          blob = await fetchSynthesisBlob({
            text: plainText, speakerId, guidance: null, engine, model, cfgValue, timesteps, loraMode: 'on', autoAnnotate: false, postProcess: dspOn, dspParams
          });
        } else if (mode === 'raw_tts') {
          // The raw variant is the A/B reference and always bypasses the module,
          // whatever the toggle says.
          blob = await fetchSynthesisBlob({
            text, speakerId, guidance, engine, model, cfgValue, timesteps, loraMode: 'on', autoAnnotate: true, postProcess: false
          });
        }
        return { mode, blob };
      });

      const results = await Promise.all(tasks);

      // Hide all player boxes initially
      if (playerBoxLoraOn) playerBoxLoraOn.classList.add('hidden');
      if (playerBoxLoraOff) playerBoxLoraOff.classList.add('hidden');
      if (playerBoxNoEmotion) playerBoxNoEmotion.classList.add('hidden');
      if (playerBoxRaw) playerBoxRaw.classList.add('hidden');

      let firstAudioToPlay = null;

      results.forEach(({ mode, blob }) => {
        if (mode === 'lora_on') {
          if (currentAudioUrlOn) URL.revokeObjectURL(currentAudioUrlOn);
          currentAudioUrlOn = URL.createObjectURL(blob);
          if (audioPlayerLoraOn) {
            audioPlayerLoraOn.src = currentAudioUrlOn;
            if (!firstAudioToPlay) firstAudioToPlay = audioPlayerLoraOn;
          }
          if (btnDownloadLoraOn) {
            btnDownloadLoraOn.href = currentAudioUrlOn;
            btnDownloadLoraOn.download = `${ts}_${spkPrefix}lora_on.wav`;
          }
          if (playerBoxLoraOn) playerBoxLoraOn.classList.remove('hidden');
        } else if (mode === 'lora_off') {
          if (currentAudioUrlOff) URL.revokeObjectURL(currentAudioUrlOff);
          currentAudioUrlOff = URL.createObjectURL(blob);
          if (audioPlayerLoraOff) {
            audioPlayerLoraOff.src = currentAudioUrlOff;
            if (!firstAudioToPlay) firstAudioToPlay = audioPlayerLoraOff;
          }
          if (btnDownloadLoraOff) {
            btnDownloadLoraOff.href = currentAudioUrlOff;
            btnDownloadLoraOff.download = `${ts}_${spkPrefix}lora_off.wav`;
          }
          if (playerBoxLoraOff) playerBoxLoraOff.classList.remove('hidden');
        } else if (mode === 'no_emotion') {
          if (currentAudioUrlNoEmotion) URL.revokeObjectURL(currentAudioUrlNoEmotion);
          currentAudioUrlNoEmotion = URL.createObjectURL(blob);
          if (audioPlayerNoEmotion) {
            audioPlayerNoEmotion.src = currentAudioUrlNoEmotion;
            if (!firstAudioToPlay) firstAudioToPlay = audioPlayerNoEmotion;
          }
          if (btnDownloadNoEmotion) {
            btnDownloadNoEmotion.href = currentAudioUrlNoEmotion;
            btnDownloadNoEmotion.download = `${ts}_${spkPrefix}no_emotion.wav`;
          }
          if (playerBoxNoEmotion) playerBoxNoEmotion.classList.remove('hidden');
        } else if (mode === 'raw_tts') {
          if (currentAudioUrlRaw) URL.revokeObjectURL(currentAudioUrlRaw);
          currentAudioUrlRaw = URL.createObjectURL(blob);
          if (audioPlayerRaw) {
            audioPlayerRaw.src = currentAudioUrlRaw;
            if (!firstAudioToPlay) firstAudioToPlay = audioPlayerRaw;
          }
          if (btnDownloadRaw) {
            btnDownloadRaw.href = currentAudioUrlRaw;
            btnDownloadRaw.download = `${ts}_${spkPrefix}raw_audio.wav`;
          }
          if (playerBoxRaw) playerBoxRaw.classList.remove('hidden');
        }
      });

      if (audioPlayerCard) {
        audioPlayerCard.classList.remove('hidden');
      }

      if (firstAudioToPlay) {
        firstAudioToPlay.play().catch(() => {});
      }

      // Also trigger text annotation render if output was empty
      if (!outputEditableText.value.trim()) {
        handleAnnotate();
      } else {
        const activeBtn = document.querySelector('.tab-btn.active');
        const activeTab = activeBtn ? activeBtn.getAttribute('data-tab') : 'editor';
        const tabEl = document.getElementById(`tab-${activeTab}`);
        if (tabEl) tabEl.classList.remove('hidden');
      }
    } catch (err) {
      if (errorBanner) {
        errorBanner.classList.remove('hidden');
        errorTitle.textContent = 'เกิดข้อผิดพลาดในการสังเคราะห์เสียง';
        errorDetailText.textContent = err.message;
      }
      if (rawJson) {
        rawJson.textContent = JSON.stringify({ error: err.message, status: "synthesis_failed" }, null, 2);
      }
      alert(`เกิดข้อผิดพลาดในการสังเคราะห์เสียง: ${err.message}`);
    } finally {
      showLoading(false);
    }
  }

  function renderResults(data, engine) {
    emptyState.classList.add('hidden');
    
    // Check fallback / error status
    if (data.fallback || data.error || data.error_detail) {
      if (errorBanner) {
        errorBanner.classList.remove('hidden');
        errorTitle.textContent = '⚠️ การประมวลผล LLM ไม่สำเร็จ (ใช้งาน Fallback Neutral)';
        errorDetailText.textContent = data.error_detail || data.error || 'โมเดลไม่ตอบสนองหรือโควตาใช้งานหมด (429 RESOURCE_EXHAUSTED)';
      }
      if (jsonStatusBadge) {
        jsonStatusBadge.textContent = 'Fallback Neutral (LLM Issue)';
        jsonStatusBadge.className = 'json-status-badge error-badge';
      }
    } else {
      if (errorBanner) {
        errorBanner.classList.add('hidden');
      }
      if (jsonStatusBadge) {
        jsonStatusBadge.textContent = 'HTTP 200 OK (Success)';
        jsonStatusBadge.className = 'json-status-badge';
      }
    }

    // Show model badge
    modelBadge.classList.remove('hidden');
    modelName.textContent = data.model_used;
    if (data.fallback) {
      fallbackIndicator.className = 'badge-tag fallback';
      fallbackIndicator.textContent = 'Fallback Neutral';
    } else {
      fallbackIndicator.className = 'badge-tag normal';
      fallbackIndicator.textContent = 'Normal';
    }

    // 1. Populate Editable Output Textarea.
    // The short "[sad] ... [happy] ..." form, not data.text: data.text is a
    // single-shot rendering carrying only the FIRST instruction, so editing and
    // re-submitting it collapsed a multi-emotion script into one tone.
    outputEditableText.value = data.script || data.text;
    updateOutputPreview();

    // Per-segment instruction view, driven by the same payload.
    lastRenderData = data;
    renderSegmentedEditor();

    // Populate Gemini Prompt if available
    if (data.prompt && engine === 'gemini') {
      geminiPromptSection.classList.remove('hidden');
      geminiPromptEditable.value = data.prompt;
    } else {
      geminiPromptSection.classList.add('hidden');
      geminiPromptEditable.value = '';
    }

    // 2. Render Segments Tab
    segmentsContainer.innerHTML = '';
    if (data.segments && data.segments.length > 0) {
      data.segments.forEach((seg, idx) => {
        const item = document.createElement('div');
        item.className = `segment-item border-${seg.tone}`;
        item.innerHTML = `
          <div class="segment-meta">
            <div class="segment-meta-left">
              <span class="seg-index">#${idx + 1}</span>
              <span class="tone-chip tone-${seg.tone}">${seg.style || seg.tone}</span>
            </div>
            <span class="intensity-stars">${formatIntensityStars(seg.intensity)}</span>
          </div>
          <div class="segment-text">${escapeHtml(seg.text)}</div>
        `;
        segmentsContainer.appendChild(item);
      });
    }

    // 3. Raw JSON Tab
    rawJson.textContent = JSON.stringify(data, null, 2);

    // Default to editor tab if nothing active
    const activeBtn = document.querySelector('.tab-btn.active');
    const activeTab = activeBtn ? activeBtn.getAttribute('data-tab') : 'editor';
    switchTab(activeTab);
  }

  // Copy helpers
  function setupCopyBtn(btn, getSourceText) {
    if (!btn) return;
    btn.addEventListener('click', async () => {
      const text = getSourceText();
      if (!text) return;
      try {
        await navigator.clipboard.writeText(text);
        const originalHtml = btn.innerHTML;
        btn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#4ade80" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>
          <span style="color:#4ade80;">คัดลอกแล้ว!</span>
        `;
        setTimeout(() => {
          btn.innerHTML = originalHtml;
        }, 1800);
      } catch (e) {
        alert('ไม่สามารถคัดลอกข้อความได้');
      }
    });
  }

  setupCopyBtn(btnCopyOutput, () => outputEditableText.value);
  setupCopyBtn(btnCopySegmented, () => segmentedEditableText.value);
  setupCopyBtn(btnCopyPrompt, () => geminiPromptEditable.value);
  setupCopyBtn(btnCopyJson, () => rawJson.textContent);

  btnProcess.addEventListener('click', handleAnnotate);
  btnSynthesizeDirect.addEventListener('click', handleSynthesize);

  // Allow Ctrl+Enter to trigger annotate or synthesize
  textInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleSynthesize();
    }
  });
  if (guidanceInput) {
    guidanceInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        handleAnnotate();
      }
    });
  }
});
