/**
 * Emotion TTS Benchmark & Pipeline Comparison Suite
 * Interactive Frontend Controller
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const healthStatus = document.getElementById('health-status');
  const speakerSelect = document.getElementById('speaker-select');
  const btnPlayRef = document.getElementById('btn-play-ref');
  const speakerRefAudio = document.getElementById('speaker-ref-audio');
  const presetPillGroup = document.getElementById('preset-pill-group');
  const testTextInput = document.getElementById('test-text-input');
  const emotionCheckboxGrid = document.getElementById('emotion-checkbox-grid');
  const btnSelectAllEmotions = document.getElementById('btn-select-all-emotions');
  const btnClearEmotions = document.getElementById('btn-clear-emotions');
  
  const repeatsPicker = document.getElementById('repeats-picker');
  const paramRepeats = document.getElementById('param-repeats');
  const paramIntensity = document.getElementById('param-intensity');
  const paramCfg = document.getElementById('param-cfg');
  const paramLoraMode = document.getElementById('param-lora-mode');
  const benchDspInputs = [...document.querySelectorAll('.bench-dsp-input')];
  const benchDspCount = document.getElementById('bench-dsp-count');
  const matrixVariantBar = document.getElementById('matrix-variant-bar');
  const mvPills = document.getElementById('mv-pills');

  // Every take result of the current run, so switching the displayed DSP variant
  // can redraw the matrix from memory instead of re-running the sampler.
  const takeResults = new Map();
  let activeVariantId = null;

  function getSelectedDspVariants() {
    const ids = benchDspInputs.filter(i => i.checked).map(i => i.value);
    return ids.length ? ids : ['emotion_on'];
  }

  function syncBenchDspPicker() {
    const n = getSelectedDspVariants().length;
    benchDspInputs.forEach((i) => {
      const opt = i.closest('.bench-dsp-opt');
      if (opt) opt.classList.toggle('active', i.checked);
      // The endpoint accepts six; four is as many as stays readable in the matrix.
      i.disabled = !i.checked && n >= 4;
      if (opt) opt.classList.toggle('is-locked', i.disabled);
    });
    if (benchDspCount) {
      benchDspCount.textContent = n === 1 ? '1 สูตร' : `${n} สูตร · gen รอบเดียว`;
      benchDspCount.classList.toggle('is-multi', n > 1);
    }
  }

  benchDspInputs.forEach(i => i.addEventListener('change', syncBenchDspPicker));
  syncBenchDspPicker();

  const summaryEmotionsCount = document.getElementById('summary-emotions-count');
  const summaryTakesCount = document.getElementById('summary-takes-count');
  const summaryTotalTakes = document.getElementById('summary-total-takes');

  const btnRunBenchmark = document.getElementById('btn-run-benchmark');
  const btnStopBenchmark = document.getElementById('btn-stop-benchmark');
  const btnExportZip = document.getElementById('btn-export-zip');
  const btnPlayAllSuite = document.getElementById('btn-play-all-suite');

  const progressCard = document.getElementById('progress-card');
  const progressTextLabel = document.getElementById('progress-text-label');
  const statCompletedCount = document.getElementById('stat-completed-count');
  const statElapsedTime = document.getElementById('stat-elapsed-time');
  const progressBarFill = document.getElementById('progress-bar-fill');

  const resultsEmptyState = document.getElementById('results-empty-state');
  const resultsTableWrapper = document.getElementById('results-table-wrapper');
  const benchmarkTable = document.getElementById('benchmark-table');
  const benchmarkTbody = document.getElementById('benchmark-tbody');
  const currentSessionTag = document.getElementById('current-session-tag');

  const btnOpenHistory = document.getElementById('btn-open-history');
  const historyModal = document.getElementById('history-modal');
  const btnCloseHistory = document.getElementById('btn-close-history');
  const historyList = document.getElementById('history-list');
  const historyLoading = document.getElementById('history-loading');

  const globalSuiteAudio = document.getElementById('global-suite-audio');

  // Application State
  let presetsData = null;
  let allEmotions = [];
  let isRunning = false;
  let shouldStop = false;
  let currentSessionId = null;
  let currentSessionData = null;
  let currentActiveAudio = null;
  let currentActivePlayBtn = null;
  let sequentialQueue = [];
  let sequentialIndex = 0;
  let timerInterval = null;
  let startTime = 0;

  // Initialize
  initPresets();
  checkHealth();

  async function checkHealth() {
    try {
      const res = await fetch('/health');
      const data = await res.json();
      if (data.status === 'ok') {
        const synth = data.synthesizer || {};
        const mode = synth.mode || 'ready';
        healthStatus.innerHTML = `<span class="status-dot"></span><span class="status-label">ระบบพร้อมทำงาน (${mode})</span>`;
      }
    } catch (e) {
      healthStatus.innerHTML = `<span class="status-dot dot-amber"></span><span class="status-label">กำลังเชื่อมต่อ Backend...</span>`;
    }
  }

  async function initPresets() {
    try {
      const res = await fetch('/api/benchmark/presets');
      presetsData = await res.json();
      allEmotions = presetsData.emotions || [];

      // 1. Populate Speaker Dropdown
      populateSpeakers(presetsData.speakers || []);

      // 2. Populate Preset Sentence Pills
      populatePresetSentences(presetsData.preset_sentences || []);

      // 3. Populate Emotion Checkboxes
      populateEmotionCheckboxes(allEmotions);

      // Set initial text
      if (presetsData.default_params && presetsData.default_params.text) {
        testTextInput.value = presetsData.default_params.text;
      }

      updateSummaryCounters();
    } catch (e) {
      console.error('Failed to load presets:', e);
    }
  }

  function populateSpeakers(speakers) {
    speakerSelect.innerHTML = '<option value="">-- ไม่ใช้เสียงโคลน (Auto-Seed Neutral Voice) --</option>';
    speakers.forEach(spk => {
      const opt = document.createElement('option');
      opt.value = spk.id;
      opt.textContent = `${spk.name || spk.id} (${spk.filename || 'ref'})`;
      speakerSelect.appendChild(opt);
    });
  }

  function populatePresetSentences(presets) {
    presetPillGroup.innerHTML = '';
    presets.forEach((p, idx) => {
      const pill = document.createElement('button');
      pill.type = 'button';
      pill.className = `preset-pill ${idx === 0 ? 'active' : ''}`;
      pill.textContent = p.title;
      pill.title = p.desc || '';
      pill.addEventListener('click', () => {
        document.querySelectorAll('.preset-pill').forEach(el => el.classList.remove('active'));
        pill.classList.add('active');
        testTextInput.value = p.text;
      });
      presetPillGroup.appendChild(pill);
    });
  }

  function populateEmotionCheckboxes(emotions) {
    emotionCheckboxGrid.innerHTML = '';
    emotions.forEach(emo => {
      const label = document.createElement('label');
      label.className = 'emotion-pill-chk active';
      label.dataset.emotion = emo.id;
      label.innerHTML = `
        <input type="checkbox" value="${emo.id}" checked>
        <span class="chk-icon">✓</span>
        <span>${emo.icon} ${emo.name_th.split('/')[0].trim()}</span>
      `;

      // A <label> wrapping its own <input> already toggles it on click, so the
      // old handler's manual flip ran in addition to the native one and the two
      // cancelled out -- every click left the checkbox exactly where it was.
      // Listening to the input's own change event is what the other pickers on
      // this page do, and it fires once per real state change.
      const input = label.querySelector('input');
      input.addEventListener('change', () => {
        label.classList.toggle('active', input.checked);
        updateSummaryCounters();
      });

      emotionCheckboxGrid.appendChild(label);
    });
  }

  function getSelectedEmotions() {
    return [...document.querySelectorAll('#emotion-checkbox-grid input[type="checkbox"]:checked')].map(i => i.value);
  }

  function updateSummaryCounters() {
    const selected = getSelectedEmotions();
    const repeats = parseInt(paramRepeats.value || '3', 10);
    summaryEmotionsCount.textContent = selected.length;
    summaryTakesCount.textContent = repeats;
    summaryTotalTakes.textContent = selected.length * repeats;

    // Toggle take header columns in table
    for (let i = 1; i <= 5; i++) {
      const th = document.getElementById(`th-take-${i}`);
      if (th) {
        th.style.display = (i <= repeats) ? '' : 'none';
      }
    }
  }

  // Speaker Ref Audio Play
  speakerSelect.addEventListener('change', () => {
    const spkId = speakerSelect.value.trim();
    if (spkId) {
      btnPlayRef.classList.remove('hidden');
      speakerRefAudio.src = `/speakers/${encodeURIComponent(spkId)}/audio`;
    } else {
      btnPlayRef.classList.add('hidden');
      speakerRefAudio.pause();
      speakerRefAudio.src = '';
    }
  });

  btnPlayRef.addEventListener('click', () => {
    if (!speakerRefAudio.src) return;
    if (speakerRefAudio.paused) {
      speakerRefAudio.play();
      btnPlayRef.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg> <span>หยุด Ref</span>`;
    } else {
      speakerRefAudio.pause();
      btnPlayRef.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> <span>ฟังเสียง Ref</span>`;
    }
  });

  speakerRefAudio.addEventListener('ended', () => {
    btnPlayRef.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> <span>ฟังเสียง Ref</span>`;
  });

  // Repeats Buttons
  repeatsPicker.querySelectorAll('.repeat-opt').forEach(btn => {
    btn.addEventListener('click', () => {
      repeatsPicker.querySelectorAll('.repeat-opt').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      paramRepeats.value = btn.dataset.repeats;
      updateSummaryCounters();
    });
  });

  // Emotion selection helpers
  btnSelectAllEmotions.addEventListener('click', () => {
    document.querySelectorAll('#emotion-checkbox-grid .emotion-pill-chk').forEach(el => {
      el.classList.add('active');
      el.querySelector('input').checked = true;
    });
    updateSummaryCounters();
  });

  btnClearEmotions.addEventListener('click', () => {
    document.querySelectorAll('#emotion-checkbox-grid .emotion-pill-chk').forEach(el => {
      el.classList.remove('active');
      el.querySelector('input').checked = false;
    });
    updateSummaryCounters();
  });

  // ---------------------------------------------------------------------------
  // Benchmark Run Execution Logic
  // ---------------------------------------------------------------------------

  btnRunBenchmark.addEventListener('click', async () => {
    const text = testTextInput.value.trim();
    if (!text) {
      alert('กรุณากรอกข้อความภาษาไทยที่ต้องการทดสอบ');
      testTextInput.focus();
      return;
    }

    const selectedEmotions = getSelectedEmotions();
    if (selectedEmotions.length === 0) {
      alert('กรุณาเลือกอย่างน้อย 1 อารมณ์เพื่อทำการทดสอบ');
      return;
    }

    const repeats = parseInt(paramRepeats.value || '3', 10);
    const speakerId = speakerSelect.value.trim() || null;
    const intensity = parseInt(paramIntensity.value || '2', 10);
    const cfgValue = parseFloat(paramCfg.value || '2.5');
    const loraMode = paramLoraMode.value || 'on';
    const dspVariants = getSelectedDspVariants();
    const postProcess = dspVariants.length === 1
      ? (window.DSP_VARIANT_SPECS[dspVariants[0]] || {}).post_process !== false
      : true;

    // Start benchmark session
    startBenchmark({
      text,
      emotions: selectedEmotions,
      repeats,
      speakerId,
      intensity,
      cfgValue,
      loraMode,
      postProcess,
      dspVariants,
    });
  });

  btnStopBenchmark.addEventListener('click', () => {
    shouldStop = true;
    btnStopBenchmark.innerHTML = `<span>กำลังหยุด...</span>`;
  });

  async function startBenchmark(config) {
    isRunning = true;
    shouldStop = false;
    btnRunBenchmark.classList.add('hidden');
    btnStopBenchmark.classList.remove('hidden');
    btnStopBenchmark.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12"></rect></svg> <span>🛑 หยุด (Stop)</span>`;
    btnExportZip.disabled = true;

    // Reset Table and Progress
    resultsEmptyState.classList.add('hidden');
    resultsTableWrapper.classList.remove('hidden');
    progressCard.classList.remove('hidden');

    startTime = Date.now();
    startTimer();

    try {
      // 1. Initialize Session via Backend API
      const initPayload = {
        speaker_id: config.speakerId,
        text: config.text,
        emotions: config.emotions,
        repeats: config.repeats,
        intensity: config.intensity,
        cfg_value: config.cfgValue,
        inference_timesteps: 10,
        lora_mode: config.loraMode,
        post_process: config.postProcess,
      };

      const initRes = await fetch('/api/benchmark/session/init', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(initPayload),
      });

      if (!initRes.ok) {
        throw new Error(`Session Init Failed: ${initRes.statusText}`);
      }

      const sessionInfo = await initRes.json();
      currentSessionId = sessionInfo.session_id;
      currentSessionTag.textContent = `Session: ${currentSessionId}`;

      // 2. Render Empty Skeleton Table Rows
      renderTableSkeleton(config.emotions, config.repeats, config.intensity);

      // 3. Build Task Queue
      const queue = [];
      config.emotions.forEach(emotion => {
        for (let takeIdx = 1; takeIdx <= config.repeats; takeIdx++) {
          queue.push({
            session_id: currentSessionId,
            emotion,
            take_idx: takeIdx,
            text: config.text,
            intensity: config.intensity,
            speaker_id: config.speakerId,
            cfg_value: config.cfgValue,
            lora_mode: config.loraMode,
            post_process: config.postProcess,
            // One ticked variant keeps the classic single-file take (and its plain
            // filename); more than one shares a generation across the treatments.
            variants: config.dspVariants.length > 1
              ? config.dspVariants.map(id => ({
                  id,
                  label: window.DSP_VARIANT_SPECS[id].label,
                  post_process: window.DSP_VARIANT_SPECS[id].post_process,
                  params: window.DSP_VARIANT_SPECS[id].params
                }))
              : null,
          });
        }
      });

      const totalTasks = queue.length;
      let completedTasks = 0;
      takeResults.clear();
      activeVariantId = null;
      if (matrixVariantBar) matrixVariantBar.classList.add('hidden');

      // 4. Process Queue Sequentially
      for (const task of queue) {
        if (shouldStop) {
          progressTextLabel.textContent = `การทดสอบถูกหยุดโดยผู้ใช้ (${completedTasks}/${totalTasks} Takes)`;
          break;
        }

        const emotionMeta = allEmotions.find(e => e.id === task.emotion) || { name_th: task.emotion, icon: '🎙️' };
        progressTextLabel.textContent = `กำลังสร้าง: [${task.emotion}] ${emotionMeta.name_th} — Take ${task.take_idx}/${config.repeats}...`;
        statCompletedCount.textContent = `${completedTasks} / ${totalTasks} Takes`;
        progressBarFill.style.width = `${Math.round((completedTasks / totalTasks) * 100)}%`;

        // Mark cell as generating
        setCellGenerating(task.emotion, task.take_idx);

        try {
          const takeRes = await fetch('/api/benchmark/run-take', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(task),
          });

          const takeResult = await takeRes.json();
          renderTakeResultCell(takeResult);
          completedTasks++;
        } catch (err) {
          console.error('Take execution error:', err);
          renderTakeErrorCell(task.emotion, task.take_idx, err.message);
          completedTasks++;
        }

        statCompletedCount.textContent = `${completedTasks} / ${totalTasks} Takes`;
        progressBarFill.style.width = `${Math.round((completedTasks / totalTasks) * 100)}%`;
      }

      // 5. Completion
      stopTimer();
      progressBarFill.style.width = '100%';
      if (!shouldStop) {
        progressTextLabel.textContent = `✨ การทดสอบเสร็จสมบูรณ์ ครบทั้ง ${totalTasks} Takes!`;
      }
      btnExportZip.disabled = false;

    } catch (e) {
      alert(`เกิดข้อผิดพลาดในการรันการทดสอบ: ${e.message}`);
      stopTimer();
    } finally {
      isRunning = false;
      btnRunBenchmark.classList.remove('hidden');
      btnStopBenchmark.classList.add('hidden');
    }
  }

  function startTimer() {
    if (timerInterval) clearInterval(timerInterval);
    timerInterval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - startTime) / 1000);
      const m = String(Math.floor(elapsed / 60)).padStart(2, '0');
      const s = String(elapsed % 60).padStart(2, '0');
      statElapsedTime.textContent = `${m}:${s}`;
    }, 1000);
  }

  function stopTimer() {
    if (timerInterval) {
      clearInterval(timerInterval);
      timerInterval = null;
    }
  }

  // ---------------------------------------------------------------------------
  // Table Rendering & Cell Updates
  // ---------------------------------------------------------------------------

  function renderTableSkeleton(emotions, repeats, intensity) {
    benchmarkTbody.innerHTML = '';

    emotions.forEach(emoId => {
      const emo = allEmotions.find(e => e.id === emoId) || {
        id: emoId,
        name_th: emoId,
        icon: '🎙️',
        color_class: 'tone-neutral',
        default_instruction: '',
      };

      const tr = document.createElement('tr');
      tr.id = `row-${emo.id}`;
      tr.dataset.emotion = emo.id;

      // Column 1: Emotion badge
      const tdEmo = document.createElement('td');
      tdEmo.innerHTML = `
        <div class="emotion-badge-cell">
          <span class="emotion-tag-badge ${emo.color_class}">${emo.icon} ${emo.id}</span>
          <small style="color:#94a3b8;font-size:11.5px;">${emo.name_th}</small>
        </div>
      `;
      tr.appendChild(tdEmo);

      // Column 2: Instruction
      const tdInstr = document.createElement('td');
      tdInstr.innerHTML = `<span class="instruction-code" id="instr-${emo.id}">${emo.default_instruction || '(Default Tone)'}</span>`;
      tr.appendChild(tdInstr);

      // Columns 3-7: Takes
      for (let i = 1; i <= 5; i++) {
        const tdTake = document.createElement('td');
        tdTake.id = `cell-${emo.id}-${i}`;
        tdTake.className = 'td-take-cell';
        if (i > repeats) {
          tdTake.style.display = 'none';
        } else {
          tdTake.innerHTML = `
            <div class="mini-take-player is-pending">
              <span style="font-size:11.5px;">Take ${i} (รอรัน)</span>
            </div>
          `;
        }
        tr.appendChild(tdTake);
      }

      // Column 8: Compare All Takes Button
      const tdCompare = document.createElement('td');
      tdCompare.innerHTML = `
        <button type="button" class="btn-compare-play" id="btn-compare-${emo.id}" title="เล่นเทียบ Take 1 -> 2 -> 3">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
          <span>เล่นเทียบ ${repeats} Takes</span>
        </button>
      `;
      tdCompare.querySelector('button').addEventListener('click', () => {
        playSequentialRow(emo.id, repeats);
      });
      tr.appendChild(tdCompare);

      // Column 9: Metrics Cell
      const tdMetrics = document.createElement('td');
      tdMetrics.id = `metrics-${emo.id}`;
      tdMetrics.innerHTML = `<div class="metrics-pill-group"><span class="metric-pill" style="color:#64748b;">—</span></div>`;
      tr.appendChild(tdMetrics);

      // Column 10: Rerun row button
      const tdAction = document.createElement('td');
      tdAction.innerHTML = `
        <button type="button" class="btn-rerun-row" title="รันซ้ำเฉพาะอารมณ์นี้">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>
          <span>รันซ้ำ</span>
        </button>
      `;
      tdAction.querySelector('button').addEventListener('click', () => {
        rerunSingleEmotion(emo.id, repeats);
      });
      tr.appendChild(tdAction);

      benchmarkTbody.appendChild(tr);
    });
  }

  function setCellGenerating(emotion, takeIdx) {
    const cell = document.getElementById(`cell-${emotion}-${takeIdx}`);
    if (cell) {
      cell.innerHTML = `
        <div class="mini-take-player is-generating">
          <div class="spinner-small"></div>
          <span style="font-size:11px;">กำลังสังเคราะห์...</span>
        </div>
      `;
    }
  }

  // The bar only earns its space when there is something to switch between.
  function renderVariantBar(variants) {
    if (!matrixVariantBar || !mvPills) return;
    if (!variants || variants.length < 2) {
      matrixVariantBar.classList.add('hidden');
      activeVariantId = variants && variants.length ? variants[0].id : null;
      return;
    }
    if (!activeVariantId || !variants.some(v => v.id === activeVariantId)) {
      activeVariantId = variants[0].id;
    }
    mvPills.innerHTML = variants.map(v =>
      `<button type="button" class="mv-pill${v.id === activeVariantId ? ' active' : ''}" data-variant="${v.id}">${v.label}</button>`
    ).join('');
    mvPills.querySelectorAll('.mv-pill').forEach((btn) => {
      btn.addEventListener('click', () => {
        activeVariantId = btn.dataset.variant;
        mvPills.querySelectorAll('.mv-pill').forEach(b => b.classList.toggle('active', b === btn));
        // Redraw every cell already collected, so the whole matrix -- players,
        // durations and the per-emotion metric pills -- describes one variant.
        takeResults.forEach(r => renderTakeResultCell(r));
      });
    });
    matrixVariantBar.classList.remove('hidden');
  }

  // Which file this cell should show: the active variant if the take carries one,
  // otherwise the take's own top-level fields (old sessions, single-variant runs).
  function pickVariant(result) {
    const list = result.variants || [];
    if (!list.length) return null;
    return list.find(v => v.id === activeVariantId) || list[0];
  }

  function renderTakeResultCell(result) {
    if (!result.error) {
      takeResults.set(`${result.emotion}_${result.take_idx}`, result);
      renderVariantBar(result.variants);
    }
    const chosen = pickVariant(result);
    const { emotion, take_idx, instruction, elapsed_s, error } = result;
    const audio_url = chosen ? chosen.audio_url : result.audio_url;
    const filename = chosen ? chosen.filename : result.filename;
    const metrics = chosen ? chosen.metrics : result.metrics;
    const cell = document.getElementById(`cell-${emotion}-${take_idx}`);
    if (!cell) return;

    if (error || !audio_url) {
      renderTakeErrorCell(emotion, take_idx, error || 'Failed');
      return;
    }

    // Update instruction text if arrived
    if (instruction) {
      const instrEl = document.getElementById(`instr-${emotion}`);
      if (instrEl) instrEl.textContent = instruction;
    }

    const dur = metrics ? `${metrics.dur_s}s` : '—';

    cell.innerHTML = `
      <div class="mini-take-player" id="player-${emotion}-${take_idx}" data-url="${audio_url}">
        <button type="button" class="btn-mini-play" title="เล่นเสียง">
          <svg class="play-svg" width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
          <svg class="pause-svg hidden" width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>
        </button>
        <div class="mini-meta">
          <span class="mini-dur">${dur}</span>
        </div>
        <a href="${audio_url}" download="${filename}" class="btn-mini-dl" title="ดาวน์โหลดไฟล์ WAV">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
        </a>
      </div>
    `;

    // Attach Play Click Listener
    const playerBox = cell.querySelector('.mini-take-player');
    const playBtn = cell.querySelector('.btn-mini-play');
    playBtn.addEventListener('click', () => {
      toggleSingleAudio(audio_url, playerBox, playBtn);
    });

    // Update Metrics in Row
    if (metrics) {
      updateRowMetrics(emotion, metrics);
    }
  }

  function renderTakeErrorCell(emotion, takeIdx, errMsg) {
    const cell = document.getElementById(`cell-${emotion}-${takeIdx}`);
    if (cell) {
      cell.innerHTML = `
        <div class="mini-take-player is-pending" style="border-color:#ef4444;color:#f87171;" title="${errMsg}">
          <span style="font-size:11px;">⚠️ ล้มเหลว</span>
        </div>
      `;
    }
  }

  function updateRowMetrics(emotion, m) {
    const metricsCell = document.getElementById(`metrics-${emotion}`);
    if (!metricsCell) return;

    metricsCell.innerHTML = `
      <div class="metrics-pill-group">
        <span class="metric-pill" title="ระดับความดังเฉลี่ย">⚡ <strong>${m.energy_dbfs}</strong> dB</span>
        <span class="metric-pill" title="Pitch กลาง (F0 Median)">🎵 <strong>${m.f0_med_hz}</strong> Hz</span>
        <span class="metric-pill" title="Dynamic Pitch Spread">〰️ <strong>${m.f0_spread_st}</strong> st</span>
      </div>
    `;
  }

  // ---------------------------------------------------------------------------
  // Audio Playback Controllers
  // ---------------------------------------------------------------------------

  function toggleSingleAudio(url, playerBox, playBtn) {
    stopSequentialTour();

    if (currentActiveAudio && currentActiveAudio.src.endsWith(url) && !currentActiveAudio.paused) {
      currentActiveAudio.pause();
      setPlayerBtnState(playBtn, false);
      playerBox.classList.remove('is-playing');
      return;
    }

    if (currentActiveAudio) {
      currentActiveAudio.pause();
      if (currentActivePlayBtn) setPlayerBtnState(currentActivePlayBtn, false);
      document.querySelectorAll('.mini-take-player').forEach(el => el.classList.remove('is-playing'));
    }

    globalSuiteAudio.src = url;
    globalSuiteAudio.play();
    currentActiveAudio = globalSuiteAudio;
    currentActivePlayBtn = playBtn;

    setPlayerBtnState(playBtn, true);
    playerBox.classList.add('is-playing');

    globalSuiteAudio.onended = () => {
      setPlayerBtnState(playBtn, false);
      playerBox.classList.remove('is-playing');
      currentActiveAudio = null;
      currentActivePlayBtn = null;
    };
  }

  function setPlayerBtnState(btn, isPlaying) {
    if (!btn) return;
    const playSvg = btn.querySelector('.play-svg');
    const pauseSvg = btn.querySelector('.pause-svg');
    if (isPlaying) {
      btn.classList.add('btn-is-playing');
      if (playSvg) playSvg.classList.add('hidden');
      if (pauseSvg) pauseSvg.classList.remove('hidden');
    } else {
      btn.classList.remove('btn-is-playing');
      if (playSvg) playSvg.classList.remove('hidden');
      if (pauseSvg) pauseSvg.classList.add('hidden');
    }
  }

  // Sequential Compare Play (A/B/C)
  function playSequentialRow(emotion, repeats) {
    stopSequentialTour();

    const row = document.getElementById(`row-${emotion}`);
    const compareBtn = document.getElementById(`btn-compare-${emotion}`);
    if (!row) return;

    // Collect URLs from row takes
    const urls = [];
    for (let i = 1; i <= repeats; i++) {
      const p = document.getElementById(`player-${emotion}-${i}`);
      if (p && p.dataset.url) {
        urls.push({
          url: p.dataset.url,
          playerBox: p,
          playBtn: p.querySelector('.btn-mini-play'),
          takeIdx: i,
        });
      }
    }

    if (urls.length === 0) {
      alert('ยังไม่มีเสียงที่สังเคราะห์เสร็จในแถวนี้');
      return;
    }

    compareBtn.classList.add('playing');
    compareBtn.innerHTML = `<span>กำลังเล่น Take 1/${urls.length}...</span>`;
    row.classList.add('row-active-highlight');

    let idx = 0;
    function playNext() {
      if (idx >= urls.length) {
        compareBtn.classList.remove('playing');
        compareBtn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> <span>เล่นเทียบ ${repeats} Takes</span>`;
        row.classList.remove('row-active-highlight');
        document.querySelectorAll('.mini-take-player').forEach(el => el.classList.remove('is-playing'));
        return;
      }

      const item = urls[idx];
      compareBtn.innerHTML = `<span>กำลังเล่น Take ${item.takeIdx}/${urls.length}...</span>`;
      document.querySelectorAll('.mini-take-player').forEach(el => el.classList.remove('is-playing'));
      item.playerBox.classList.add('is-playing');

      globalSuiteAudio.src = item.url;
      globalSuiteAudio.play();

      globalSuiteAudio.onended = () => {
        item.playerBox.classList.remove('is-playing');
        idx++;
        setTimeout(playNext, 250); // slight micro-pause between takes
      };
    }

    playNext();
  }

  function stopSequentialTour() {
    globalSuiteAudio.pause();
    globalSuiteAudio.onended = null;
    document.querySelectorAll('.btn-compare-play').forEach(btn => {
      btn.classList.remove('playing');
    });
    document.querySelectorAll('.mini-take-player').forEach(el => el.classList.remove('is-playing'));
    document.querySelectorAll('tr').forEach(r => r.classList.remove('row-active-highlight'));
  }

  // Sequential Play All Tour
  btnPlayAllSuite.addEventListener('click', () => {
    stopSequentialTour();
    const allFirstTakes = [];
    document.querySelectorAll('tbody tr').forEach(row => {
      const p = row.querySelector('.mini-take-player[data-url]');
      if (p && p.dataset.url) {
        allFirstTakes.push({
          url: p.dataset.url,
          playerBox: p,
          row: row,
        });
      }
    });

    if (allFirstTakes.length === 0) {
      alert('ยังไม่มีเสียงที่สังเคราะห์เสร็จ');
      return;
    }

    let idx = 0;
    btnPlayAllSuite.innerHTML = `<span>กำลังเล่นทัวร์ (1/${allFirstTakes.length})...</span>`;

    function playNextTour() {
      if (idx >= allFirstTakes.length) {
        btnPlayAllSuite.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> <span>เล่นทุกอารมณ์ต่อเนื่อง (Tour)</span>`;
        document.querySelectorAll('.mini-take-player').forEach(el => el.classList.remove('is-playing'));
        document.querySelectorAll('tr').forEach(r => r.classList.remove('row-active-highlight'));
        return;
      }

      const item = allFirstTakes[idx];
      btnPlayAllSuite.innerHTML = `<span>กำลังเล่น (${idx + 1}/${allFirstTakes.length})...</span>`;
      document.querySelectorAll('.mini-take-player').forEach(el => el.classList.remove('is-playing'));
      document.querySelectorAll('tr').forEach(r => r.classList.remove('row-active-highlight'));

      item.playerBox.classList.add('is-playing');
      item.row.classList.add('row-active-highlight');

      globalSuiteAudio.src = item.url;
      globalSuiteAudio.play();

      globalSuiteAudio.onended = () => {
        item.playerBox.classList.remove('is-playing');
        item.row.classList.remove('row-active-highlight');
        idx++;
        setTimeout(playNextTour, 300);
      };
    }

    playNextTour();
  });

  // Re-run single emotion row
  async function rerunSingleEmotion(emotion, repeats) {
    if (isRunning) {
      alert('กรุณารอการทดสอบปัจจุบันเสร็จสิ้นก่อน');
      return;
    }

    if (!currentSessionId) {
      alert('ยังไม่มี Session ที่กำลังใช้งาน');
      return;
    }

    const text = testTextInput.value.trim();
    const speakerId = speakerSelect.value.trim() || null;
    const intensity = parseInt(paramIntensity.value || '2', 10);
    const cfgValue = parseFloat(paramCfg.value || '2.5');
    const loraMode = paramLoraMode.value || 'on';
    const dspVariants = getSelectedDspVariants();
    const postProcess = dspVariants.length === 1
      ? (window.DSP_VARIANT_SPECS[dspVariants[0]] || {}).post_process !== false
      : true;

    for (let takeIdx = 1; takeIdx <= repeats; takeIdx++) {
      setCellGenerating(emotion, takeIdx);
      try {
        const res = await fetch('/api/benchmark/run-take', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: currentSessionId,
            emotion,
            take_idx: takeIdx,
            text,
            intensity,
            speaker_id: speakerId,
            cfg_value: cfgValue,
            lora_mode: loraMode,
            post_process: postProcess,
            // Re-running one row must produce the same set of treatments as the
            // rest of the matrix, or the variant switcher would find gaps.
            variants: dspVariants.length > 1
              ? dspVariants.map(id => ({
                  id,
                  label: window.DSP_VARIANT_SPECS[id].label,
                  post_process: window.DSP_VARIANT_SPECS[id].post_process,
                  params: window.DSP_VARIANT_SPECS[id].params
                }))
              : null,
          }),
        });
        const data = await res.json();
        renderTakeResultCell(data);
      } catch (err) {
        renderTakeErrorCell(emotion, takeIdx, err.message);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // History & Export
  // ---------------------------------------------------------------------------

  btnExportZip.addEventListener('click', () => {
    if (!currentSessionId) return;
    window.location.href = `/api/benchmark/export/${currentSessionId}`;
  });

  btnOpenHistory.addEventListener('click', async () => {
    historyModal.classList.remove('hidden');
    historyLoading.classList.remove('hidden');
    historyList.innerHTML = '';

    try {
      const res = await fetch('/api/benchmark/sessions');
      const sessions = await res.json();
      historyLoading.classList.add('hidden');

      if (sessions.length === 0) {
        historyList.innerHTML = '<p style="color:#64748b;text-align:center;padding:20px;">ยังไม่มีประวัติการทดสอบ</p>';
        return;
      }

      sessions.forEach(sess => {
        const item = document.createElement('div');
        item.className = 'history-item';
        const dateStr = sess.created_at ? new Date(sess.created_at).toLocaleString('th-TH') : '—';
        const spk = sess.speaker_id || 'Base Seed Voice';
        const takes = `${sess.completed_takes || 0}/${sess.total_takes || 0} Takes`;

        item.innerHTML = `
          <div class="history-meta">
            <span class="history-title">${sess.name}</span>
            <span class="history-sub">📅 ${dateStr} · 🎙️ ${spk} · ⚡ ${takes}</span>
            <span class="history-text-snippet">"${sess.text || ''}"</span>
          </div>
          <button type="button" class="btn-load-history" data-session="${sess.session_id}">
            โหลดผลเทสนี้
          </button>
        `;

        item.querySelector('.btn-load-history').addEventListener('click', () => {
          loadSessionDetails(sess.session_id);
          historyModal.classList.add('hidden');
        });

        historyList.appendChild(item);
      });
    } catch (e) {
      historyLoading.classList.add('hidden');
      historyList.innerHTML = `<p style="color:#f87171;text-align:center;">โหลดประวัติไม่สำเร็จ: ${e.message}</p>`;
    }
  });

  btnCloseHistory.addEventListener('click', () => {
    historyModal.classList.add('hidden');
  });

  historyModal.addEventListener('click', (e) => {
    if (e.target === historyModal) historyModal.classList.add('hidden');
  });

  async function loadSessionDetails(sessionId) {
    try {
      const res = await fetch(`/api/benchmark/sessions/${sessionId}`);
      if (!res.ok) throw new Error('Session not found');
      const data = await res.json();

      currentSessionId = data.session_id;
      currentSessionTag.textContent = `Session: ${currentSessionId}`;

      // Update form inputs to match session
      if (data.text) testTextInput.value = data.text;
      if (data.speaker_id) speakerSelect.value = data.speaker_id;
      if (data.repeats) {
        paramRepeats.value = data.repeats;
        repeatsPicker.querySelectorAll('.repeat-opt').forEach(b => {
          b.classList.toggle('active', b.dataset.repeats === String(data.repeats));
        });
      }

      const emotions = data.emotions || [];
      const repeats = data.repeats || 3;
      const intensity = (data.params && data.params.intensity) || 2;

      resultsEmptyState.classList.add('hidden');
      resultsTableWrapper.classList.remove('hidden');

      renderTableSkeleton(emotions, repeats, intensity);
      takeResults.clear();
      activeVariantId = null;
      if (matrixVariantBar) matrixVariantBar.classList.add('hidden');

      const takes = data.takes || {};
      Object.values(takes).forEach(takeRecord => {
        renderTakeResultCell({
          session_id: currentSessionId,
          emotion: takeRecord.emotion,
          take_idx: takeRecord.take_idx,
          instruction: takeRecord.instruction,
          audio_url: takeRecord.audio_url,
          filename: takeRecord.filename,
          metrics: takeRecord.metrics,
          // Sessions recorded before variants existed simply have none, and the
          // cell falls back to the take's own fields.
          variants: takeRecord.variants || [],
          elapsed_s: takeRecord.elapsed_s,
          error: takeRecord.error,
        });
      });

      btnExportZip.disabled = false;
    } catch (e) {
      alert(`ไม่สามารถโหลดผลเทสได้: ${e.message}`);
    }
  }
});

/* ==========================================================================
   Fair A/B — one generation, several assemblies
   ==========================================================================
   The benchmark above runs the sampler once per take, which is right for asking
   "how consistent is this emotion". It is the wrong tool for asking "did the
   post-processing help", because two runs differ by the sampler as well as by
   the treatment. This section calls /synthesize/ab, which renders once and
   assembles the same chunks every way asked for.
   ========================================================================== */
window.DSP_VARIANT_SPECS = {
    emotion_on: {
      label: '🎚️ เปิดชั้นอารมณ์',
      post_process: true,
      params: null
    },
    cleanup: {
      label: '🚫 ปิดชั้นอารมณ์',
      post_process: true,
      params: { match_energy: false, match_rate: false, gap_emotion_s: 0.2 }
    },
    raw: {
      label: '🎵 ปิด DSP ทั้งหมด',
      post_process: false,
      params: null
    },
    dramatic: {
      label: '🎭 เว้นจังหวะเยอะ',
      post_process: true,
      params: {
        gap_same_tone_s: 0.30, gap_emotion_s: 0.70, gap_paragraph_s: 1.60,
        energy_match: 0.55, max_stretch: 0.20
      }
    },
  narration: {
    label: '📖 พูดกระชับ',
    post_process: true,
    params: {
      gap_same_tone_s: 0.14, gap_emotion_s: 0.30, gap_paragraph_s: 0.70,
      energy_match: 0.85, max_stretch: 0.12
    }
  }
};

(function () {
  const VARIANT_SPECS = window.DSP_VARIANT_SPECS;

  const TONE_COLORS = {
    neutral: 'var(--tone-neutral)', sad: 'var(--tone-sad)', happy: 'var(--tone-happy)',
    angry: 'var(--tone-angry)', excited: 'var(--tone-excited)', calm: 'var(--tone-calm)',
    nervous: 'var(--tone-nervous)', sarcastic: 'var(--tone-sarcastic)',
    scared: 'var(--tone-scared)', tired: 'var(--tone-tired)'
  };

  const TONE_TH = {
    neutral: 'เฉยๆ', sad: 'เศร้า', happy: 'มีความสุข', angry: 'โกรธ', excited: 'ตื่นเต้น',
    calm: 'สงบ', nervous: 'ประหม่า', sarcastic: 'ประชด', scared: 'กลัว', tired: 'เหนื่อย'
  };

  const $ = (id) => document.getElementById(id);

  const grid = $('ab-variant-grid');
  const btnRun = $('btn-run-ab');
  const abCount = $('ab-count');
  const abStatus = $('ab-status');
  const abStatusText = $('ab-status-text');
  const abError = $('ab-error');
  const abResults = $('ab-results');
  const abTableBody = $('ab-table-body');
  const abTimelines = $('ab-timelines');
  const abLegend = $('ab-legend');
  const abResultsNote = $('ab-results-note');
  if (!grid || !btnRun) return;

  const inputs = [...grid.querySelectorAll('.ab-variant-input')];

  function selected() {
    return inputs.filter(i => i.checked).map(i => i.value);
  }

  function syncPicker() {
    const n = selected().length;
    inputs.forEach((i) => {
      const card = i.closest('.ab-variant');
      if (card) card.classList.toggle('active', i.checked);
      // Four is the useful ceiling for listening back-to-back, and the endpoint
      // caps at six. Block the fifth rather than fail the request.
      i.disabled = !i.checked && n >= 4;
      if (card) card.classList.toggle('ab-variant-locked', i.disabled);
    });
    if (abCount) abCount.textContent = String(n);
    btnRun.disabled = n < 2;
    btnRun.title = n < 2 ? 'ต้องเลือกอย่างน้อย 2 สูตรถึงจะเทียบได้' : '';
  }

  inputs.forEach(i => i.addEventListener('change', syncPicker));
  syncPicker();

  // Echo the settings this section borrows from the benchmark card, so it is
  // never a mystery which voice and CFG the comparison actually ran at.
  function syncEcho() {
    const spk = $('speaker-select');
    const cfg = $('param-cfg');
    const lora = $('param-lora-mode');
    const txt = $('test-text-input');
    const set = (id, v) => { const el = $(id); if (el) el.textContent = v; };
    set('ab-echo-speaker', '🎙️ ' + (spk && spk.value
      ? (spk.options[spk.selectedIndex] || {}).text || spk.value
      : 'Auto-Seed Neutral'));
    set('ab-echo-cfg', 'CFG ' + (cfg ? cfg.value : '2.5'));
    set('ab-echo-lora', 'LoRA ' + (lora ? lora.value : 'on'));
    const t = txt && txt.value.trim();
    set('ab-echo-text', t ? `“${t.length > 48 ? t.slice(0, 48) + '…' : t}”` : 'ยังไม่ได้ใส่ข้อความ');
  }
  ['speaker-select', 'param-cfg', 'param-lora-mode'].forEach((id) => {
    const el = $(id);
    if (el) el.addEventListener('change', syncEcho);
  });
  const textEl = $('test-text-input');
  if (textEl) textEl.addEventListener('input', syncEcho);
  document.addEventListener('click', () => setTimeout(syncEcho, 60));
  syncEcho();

  function fmtDelta(value, unit, betterHigher) {
    if (value === null || value === undefined || Math.abs(value) < 0.005) {
      return '<span class="ab-delta ab-delta-zero">±0</span>';
    }
    const sign = value > 0 ? '+' : '';
    const cls = betterHigher === null ? 'ab-delta-neutral'
      : ((value > 0) === betterHigher ? 'ab-delta-up' : 'ab-delta-down');
    return `<span class="ab-delta ${cls}">${sign}${value.toFixed(2)}${unit}</span>`;
  }

  function renderResults(data) {
    const variants = data.variants || [];
    const base = variants[0];

    abResultsNote.textContent =
      `สร้างเสียง 1 รอบ (${data.chunk_count} ท่อน) → ประมวลผล ${variants.length} สูตร · run ${data.run_id}`;

    // --- summary table -----------------------------------------------------
    abTableBody.innerHTML = variants.map((v, idx) => {
      const isBase = idx === 0;
      const dDur = isBase ? null : v.dur_s - base.dur_s;
      const dLvl = (isBase || v.level_spread_db == null || base.level_spread_db == null)
        ? null : v.level_spread_db - base.level_spread_db;
      const dPace = (isBase || v.pace_spread_pct == null || base.pace_spread_pct == null)
        ? null : v.pace_spread_pct - base.pace_spread_pct;
      return `
        <tr class="${isBase ? 'ab-row-base' : ''}">
          <td>
            <span class="ab-row-label">${v.label}</span>
            ${isBase ? '<span class="ab-base-tag">ตัวตั้งต้น</span>' : ''}
          </td>
          <td>${v.dur_s.toFixed(2)} วิ ${dDur === null ? '' : fmtDelta(dDur, ' วิ', null)}</td>
          <td>${v.level_spread_db == null ? '—' : v.level_spread_db.toFixed(2) + ' dB'} ${dLvl === null ? '' : fmtDelta(dLvl, ' dB', true)}</td>
          <td>${v.pace_spread_pct == null ? '—' : v.pace_spread_pct.toFixed(1) + '%'} ${dPace === null ? '' : fmtDelta(dPace, '%', null)}</td>
          <td><audio class="ab-audio" controls preload="none" src="${v.audio_url}"></audio></td>
          <td><a class="ab-dl" href="${v.audio_url}" download="${data.run_id}_${v.filename}">⬇ WAV</a></td>
        </tr>`;
    }).join('');

    // --- timelines ---------------------------------------------------------
    // One shared time axis and one shared dB axis, or the rows would not be
    // comparable by eye -- which is the entire point of drawing them.
    const maxDur = Math.max(...variants.map(v => v.dur_s), 0.001);
    const allLevels = variants.flatMap(v => v.chunks.map(c => c.level_db)).filter(x => x != null);
    const loud = allLevels.length ? Math.max(...allLevels) : 0;
    const quiet = allLevels.length ? Math.min(...allLevels) : -1;
    const range = Math.max(loud - quiet, 1);

    const tones = [...new Set(variants.flatMap(v => v.chunks.map(c => c.tone || 'neutral')))];
    abLegend.innerHTML = tones.map(t =>
      `<span class="ab-legend-item"><i style="background:${TONE_COLORS[t] || 'var(--tone-neutral)'}"></i>${TONE_TH[t] || t}</span>`
    ).join('') + '<span class="ab-legend-item ab-legend-gap"><i class="ab-legend-gapbox"></i>ความเงียบ</span>';

    abTimelines.innerHTML = variants.map((v) => {
      const bars = v.chunks.map((c) => {
        const left = (c.start_s / maxDur) * 100;
        const width = Math.max(((c.end_s - c.start_s) / maxDur) * 100, 0.6);
        const h = c.level_db == null ? 40 : 28 + ((c.level_db - quiet) / range) * 72;
        const tone = c.tone || 'neutral';
        const tip = `${TONE_TH[tone] || tone} · ${(c.end_s - c.start_s).toFixed(2)} วิ`
          + (c.level_db == null ? '' : ` · ${c.level_db.toFixed(1)} dB`)
          + (c.pace_s_per_char ? ` · ${(c.pace_s_per_char * 1000).toFixed(0)} ms/ตัวอักษร` : '');
        return `<span class="ab-bar" style="left:${left}%;width:${width}%;height:${h}%;background:${TONE_COLORS[tone] || 'var(--tone-neutral)'}" title="${tip}"></span>`;
      }).join('');
      return `
        <div class="ab-timeline-row">
          <div class="ab-timeline-label">${v.label}</div>
          <div class="ab-timeline-track">${bars}</div>
          <div class="ab-timeline-dur">${v.dur_s.toFixed(2)}s</div>
        </div>`;
    }).join('');

    abResults.classList.remove('hidden');
  }

  btnRun.addEventListener('click', async () => {
    const txt = $('test-text-input');
    const text = txt ? txt.value.trim() : '';
    if (!text) {
      abError.textContent = 'กรุณาใส่ข้อความทดสอบในข้อ 1 ก่อน';
      abError.classList.remove('hidden');
      if (txt) txt.focus();
      return;
    }
    const ids = selected();
    if (ids.length < 2) return;

    const spk = $('speaker-select');
    const cfg = $('param-cfg');
    const lora = $('param-lora-mode');

    abError.classList.add('hidden');
    abResults.classList.add('hidden');
    abStatus.classList.remove('hidden');
    abStatusText.textContent = `กำลังสร้างเสียง 1 รอบ แล้วประมวลผล ${ids.length} สูตร...`;
    btnRun.disabled = true;

    try {
      const res = await fetch('/synthesize/ab', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          speaker_id: spk && spk.value ? spk.value : null,
          cfg_value: cfg ? parseFloat(cfg.value) : 2.5,
          lora_mode: lora ? lora.value : 'on',
          auto_annotate: true,
          variants: ids.map(id => ({
            id,
            label: VARIANT_SPECS[id].label,
            post_process: VARIANT_SPECS[id].post_process,
            params: VARIANT_SPECS[id].params
          }))
        })
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      renderResults(await res.json());
    } catch (e) {
      abError.textContent = `สร้างไม่สำเร็จ: ${e.message}`;
      abError.classList.remove('hidden');
    } finally {
      abStatus.classList.add('hidden');
      syncPicker();
    }
  });
})();
