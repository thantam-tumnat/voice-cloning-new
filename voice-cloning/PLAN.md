# SiangTTS — Execution Plan & Decision Tree

Roadmap + "what to do for each result." Companion to [`RESEARCH.md`](RESEARCH.md)
(VoxCPM/JaiTTS analysis) and [`RESEARCH2.md`](RESEARCH2.md) (OmniVoice analysis).
Status date: 2026-06-11.

## Where we are

- Implementation complete against voxcpm 2.0.3 (trainer, dataset bridge,
  inference, eval). 45 tests + CPU dry-run pass. Committed through `6dc9db4`.
- Nothing has run on GPU yet — the 3090 is shared with an IndexTTS job;
  coordinate before any GPU step. LoRA training (~17–20 GB) cannot share the card.

## Reference numbers (what to compare against)

| System | Thai CER short | Thai CER long | Notes |
|---|---|---|---|
| JaiTTS-v1.0 (closed) | 1.94% | 2.55% | north star; 10k h full SFT |
| Human ground truth | 1.98% | 2.47% | from JaiTTS paper |
| OmniVoice base | 7.71% (FLEURS) | — | worse than its GT 6.98% |
| VoxCPM2 vanilla | **measure in step 1** | | our baseline |

Realistic target for LoRA @ ~543 h: clearly beat vanilla VoxCPM2 and OmniVoice;
JaiTTS parity is not expected. CER is measured with
`typhoon-ai/typhoon-whisper-large-v3` (same family JaiTTS used), SIM with WavLM
x-vectors — absolute numbers from other papers aren't directly comparable, only
*our own* baseline-vs-adapter deltas are.

## Phase 0 — before training (GPU needed, light)

1. **Baseline eval** of vanilla VoxCPM2 (~8 GB VRAM, fits alongside IndexTTS if
   it stays under ~14 GB):
   `uv run python -m src.eval --prompts eval/prompts_short.tsv --cer` (and
   `prompts_long`, `prompts_digits`). Record CER/SIM — this is the "before".
2. **Data prep** (CPU/disk-bound, can run anytime):
   `prepare_vaja_thai.py` + `prepare_libritts.py`, then
   `uv run voxcpm validate -m data/vaja/train.jsonl --sample-rate 16000`.
3. **Smoke run** — tier-1-only manifest, 1 epoch (~8–12 h on the 3090). Checks:
   peak VRAM < 22 GB (see `training_summary.json`), generation *terminates*
   (trailing-silence gotcha), TB audio snapshots intelligible by ~5k steps.

**Smoke-run triage:**
- OOM → halve `batch_size` (2→1) and double `gradient_accumulation_steps`; or
  drop `max_batch_tokens` 4096→3072.
- Audio never terminates / babbles → inspect manifest for trailing silence;
  re-run prep with stricter trim. Check loss/stop is decreasing.
- Loss flat after warmup → check LR actually ramps in TB; verify LoRA params
  are the only `requires_grad` (printed at startup).

## Phase 1 — full LoRA run

Full mix (Vaja tiers 1–3 ×weights + LibriTTS-R @ 0.25), `conf/voxcpm_lora.yaml`,
3 epochs ≈ 3–5 days on the 3090. Eval every checkpoint family vs Phase-0 baseline
on three axes:

- **CER** (short + long prompts) — pronunciation/intelligibility
- **SIM** — voice-cloning fidelity (needs `ref_audio` prompts)
- **Digit-eval** (`prompts_digits.tsv`) — the augmentation's target

## Phase 2 — decision tree by eval outcome

### A. CER good (≫ baseline, ideally <4–5%), SIM good → ship it
Publish via `train/publish_to_hf.py` (license: CC-BY-NC-SA with full Vaja;
drop `tsync2`+`gigaspeech2` slices for CC-BY-SA). Write model card from
`training_summary.json`. Done — iterate only if a real use case complains.

### B. CER weak (pronunciation/tone errors), SIM fine → capacity/data-quality path
More scraped data will NOT fix this. In order, cheapest first:
1. **Train longer** if val loss was still falling at 3 epochs (extend 2–3 more).
2. **Bigger LoRA**: `r: 128, alpha: 256, enable_proj: true` (~1–2 GB more VRAM,
   one-line change). Re-run Phase 1.
3. **Data quality pass**: drop Vaja tier 3 (train tiers 1–2 only), re-run — if
   CER *improves* with less data, quality was the binding constraint.
4. **Full SFT on rented A100-80G** (~$80–200 for 2–4 days):
   `conf/voxcpm_sft.yaml`, unchanged manifests. This is the JaiTTS recipe at
   our data scale.

### C. SIM weak (clones drift / out-of-domain voices fail), CER fine → diversity path
This is the *only* scenario where GigaSpeech 2 enters:
1. Write/run `train/prepare_gigaspeech2.py`: stream a random ~1,000 h subset of
   the **refined** Thai split, gate by DNSMOS + 1–30 s window + silence trim,
   cap at ~200–300 h (~30–40 GB disk). 16 kHz native = encoder rate, no loss.
2. Add as third source at `weight: 0.3` — minority of each batch so YouTube
   acoustics don't become the output target.
3. Re-run Phase 1. If SIM improves but CER degrades, lower weight to 0.15.
4. License note: GigaSpeech 2 is non-commercial research → checkpoint becomes NC.

### D. Digit-eval weak, rest fine → augmentation tuning
1. Confirm augmentation fired: grep train logs / spot-check a few train samples.
2. Raise `thai_digit.p_full` 0.4→0.6; consider adding mixed Thai+digit forms.
3. Accept partial: phone-number digit-by-digit recital is expected to stay weak
   (RESEARCH.md §8.6.4); cardinal years/prices are the pass bar.

### E. Everything weak (barely better than baseline) → step back
1. Verify the adapter actually loaded at eval (`--adapter` path; compare with
   `--base-only` — outputs must differ).
2. Listen to TB audio snapshots across steps — if step-1k ≈ step-30k, training
   never bit: check trainable-param printout, LR curve, loss curves.
3. If training demonstrably worked but Thai is still poor, LoRA capacity is the
   suspect → jump to B.4 (full SFT) rather than iterating adapters.

### F. Full SFT (B.4) also plateaus → data ceiling
At that point the bottleneck is the 543 h corpus, not the method:
- Filtered GigaSpeech 2 at SFT scale (1–2k h, full filter pass incl. Whisper
  transcript re-validation on the rented GPU).
- Domain verticals à la JaiTTS (podcast scrapes with own ASR pipeline) — big
  project, only if SiangTTS is becoming a serious product.

## Standing constraints

- 3090 is shared with IndexTTS — check `nvidia-smi` before every GPU step.
- Torch must stay on cu128 wheels (driver 575.x / CUDA 12.9 can't run cu130).
- Input audio is encoded at 16 kHz, output generated at 48 kHz (VoxCPM2 design)
  — never "upgrade" manifests to higher rates, it buys nothing.
- Checkpoints inherit the most restrictive data license (see README §License).
