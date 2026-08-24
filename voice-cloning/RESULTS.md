# SiangTTS — Results Log

Append-only record of eval numbers. CER = Typhoon-Whisper-Large-v3 character
error rate (whitespace/punct stripped); SIM = WavLM x-vector cosine (gen vs ref).
See [`PLAN.md`](PLAN.md) for what each outcome means and what to do next.

| Date | Checkpoint | Eval set | CER | SIM | Notes |
|---|---|---|---|---|---|
| 2026-06-15 | vanilla VoxCPM2 (base, no LoRA) | prompts_short (5) | **5.70%** | — | Phase-0 baseline. No ref audio. Small sample (5 prompts) — high variance, treat as ballpark. Already < OmniVoice 7.71%; confirms VoxCPM2 has usable Thai priors. |
| 2026-06-15 | smoke LoRA, step 500 (tier-1 porjai only, ~2.7 ep) | prompts_short (5) | **3.80%** | — | Validates adapter-eval path (384 LoRA params loaded). Beats baseline (5.70%→3.80%) from a tiny single-speaker smoke. Directional only. |
| 2026-06-16 | **Phase-1 v0, 1 epoch (step 6605)** | prompts_short (5) | **0.00%** | — | All 5 short prompts transcribe character-perfect (verified ref==hyp). 5.70%→0.00%. 5 prompts + Whisper error = not literally perfect TTS, but clearly intelligible. |
| 2026-06-16 | Phase-1 v0 | prompts_digits (10) | 70% (artifact) | — | **Metric artifact, not a failure.** Model reads numerals correctly (1923→"หนึ่งพันเก้าร้อยยี่สิบสาม", phone→digit-recital). CER high only because ref has Arabic digits vs spoken Thai words. Number augmentation works. |
| 2026-06-16 | Phase-1 v0 | prompts_long (2) | 20.5% | — | long_002=0.03 (clean). long_001=0.38 — speaks full text correctly then **fails to stop, appends hallucinated speech** (termination issue, not pronunciation). |

## Cloning vs ground truth (2026-06-17, 80 prompts, macro-averaged)

Added GT (the real recording) as the ceiling/reference. Each source vs the same
reference clip / prompt text:

| Source | CER ↓ | SIM vs ref ↑ |
|---|---|---|
| GT (real recording) | 0.97% | 0.9125 |
| Base VoxCPM2 | 3.26% | 0.9057 |
| **SiangTTS v1** | **0.84%** | **0.9093** |

- SiangTTS cloning **CER (0.84%) ≤ GT (0.97%)** — as intelligible as real recordings
  (clean ASR-friendly synthesis + the ASR-judge floor; not "better than humans").
- SiangTTS **SIM (0.909) ≈ GT ceiling (0.913)** — cloning is about as close as two
  genuine recordings of the same speaker. Base is also near-ceiling.
- (Macro per-prompt averaging here vs corpus-level in eval.py gives slightly
  different CER — base 3.26% macro vs 2.96% corpus; same ranking.)

## SIM gap confirmation — it was noise (2026-06-17)

The 20-prompt SIM (base 0.905 > LoRA 0.882) suggested a cloning regression. Re-ran
on **80 prompts** (`eval/prompts_clone_large.tsv`), paired per-prompt:

| Cloning (80 prompts) | Base | SiangTTS v1 |
|---|---|---|
| SIM | 0.906 | 0.909 |
| CER | 2.96% | **0.92%** |

Paired SIM diff (LoRA − base) = **+0.0036**, 95% CI **[−0.015, +0.022]** → straddles
zero, **statistically tied**. The earlier deficit was small-sample noise. CER is
clearly better for the LoRA (~3× lower). **Conclusion: no SIM gap to fix** — the
LM-only and speaker-consistency-loss experiments are unnecessary. SiangTTS matches
base speaker similarity and substantially improves intelligibility.

## Honest base-vs-LoRA reassessment (2026-06-17)

While building the demo I measured the **base** VoxCPM2 on long-form, numerals,
and cloning — axes I'd earlier assumed (not measured) the base was weak on. It
is not. Corrected, evidence-based comparison:

| Metric (n) | Base VoxCPM2 | SiangTTS v1 |
|---|---|---|
| Short Thai CER (5) | 5.7% | 3.8% |
| Long-form CER (2) | 2.7% (clean, no runaway) | 1.6% |
| Cloning CER (20) | 5.3% | 2.5% |
| Cloning SIM (20) | 0.905 | 0.882 |
| Reads Arabic numerals | yes (verified) | yes |

**Takeaways / corrections to earlier claims in this file:**
- The "base long-form = runaway" line below was wrong — base handles long-form
  cleanly (CER 0.00 / 0.05 on two ~14 s prompts). The runaway was a **v0 (1-epoch
  LoRA)** artifact, fixed by epoch 2 — not a base behaviour.
- "Base weak on numerals" was wrong — base reads 1990/1,250/phone/times correctly
  (VoxCPM2 has built-in number handling).
- Base voice-cloning SIM (0.905) is actually ≥ LoRA (0.882) — SiangTTS does **not**
  improve speaker similarity; it improves **intelligibility** (CER ~halved on
  cloning, lower on short/long).
- Net honest story: VoxCPM2 base is already a solid Thai speaker; SiangTTS is a
  clarity/CER refinement at LoRA scale, SIM on par. Small eval sets → directional.

## Phase-1 v1 (epoch 2, step 13210) — 2026-06-16

Resumed v0 → epoch 2. Final val loss 0.908 (≈ v0; dipped to 0.89 mid-epoch).

| Eval set | v0 (1 ep) | **v1 (2 ep)** |
|---|---|---|
| prompts_short (5) | 0.00% | 3.80% |
| prompts_long (2) | 20.54% | **1.62%** |
| long_001 (runaway case) | 0.378 (hallucinated tail) | **0.000, stops at 15.4 s** |
| **cloning (20 prompts, SIM / CER)** | — | **0.882 / 2.49%** |

**Voice cloning (v1):** zero-shot from a same-speaker Common Voice ref clip,
SIM (WavLM x-vector cosine) = **0.882**, CER = 2.49%. Strong voice match without
sacrificing intelligibility (cf. OmniVoice Thai SIM-o 0.841). Eval set:
`eval/prompts_clone.tsv` (20 cv val ref-pairs). Samples in `eval/out/v1_clone/`.

**Epoch 2 fixed the long-form termination/runaway** (the v0 gap). long_001 now
speaks the full sentence and stops cleanly; long-form avg 20.5%→1.6%. Cost: short
ticked 0%→3.8% (within 5-prompt noise, still < 5.70% baseline). **v1 is the better
overall model — long-form is now usable.** This is a strong Phase-1 result: on a
single RTX 3090, short Thai ~0–4% CER, long-form ~1.6%, correct numeral reading,
working termination — from a 5.70% base.

Open: eval sets are small (5/2/10 prompts); SIM (cloning) not yet measured;
digit CER metric needs number-normalization.

**ASR-judge ceiling (confirmed 2026-06-17 by listening):** clone_08's archaic
Pali phrase "ไหว้พระอาจารีย์…" was pronounced *correctly* by the model, but
Typhoon-Whisper mis-transcribed "อาจารีย์"→"อัจฉริ", inflating that clip's CER to
16.7%. So reported CER is an **upper bound on error** — true TTS quality is at
least as good, better on rare/archaic vocabulary the ASR itself can't recognize.
Trust human listening over CER for edge-case words.

## v0 verdict (2026-06-16)

**Strong on intelligibility + digit handling.** Short Thai character-perfect;
numerals read correctly (cardinals + phone digit-recital — the JaiTTS behavior
the augmentation targeted). **One real gap: long-form generation sometimes does
not terminate** — speaks the sentence correctly, then hallucinates extra audio
(1 of 2 long prompts). This is a stop-token / runaway issue (RESEARCH §4.D), not
pronunciation. val loss was still falling at epoch end → epoch 2 may help.

Eval-method note: digit CER needs ref↔hyp number normalization to be meaningful
(convert Arabic→Thai words, or vice-versa, before CER). Current digit CER is
not a valid quality signal.

## Phase-1 run v0 (2026-06-16) — in progress

3-source mix: commonvoice (40 h, 4,245 spk, ref-paired) + porjai (clean, no ref) +
LibriTTS (EN), weights 1.0 / 0.7 / 0.42 → ~80/20 Thai/EN. 1 epoch, ~6,600 steps.

- **Step 1000:** peak VRAM **18.9 GB**, val/loss_total 0.947, snapshot durations
  2.7–5.1 s (healthy, no collapse — diverse data avoids the smoke's stop-overfit).

### OOM lesson (cost two failed launches)

VoxCPM2's flow-matching **DiT memory scales with audio frames, not LM tokens**.
At fps=25/patch=4 a 30 s clip is only ~187 packed tokens, so `max_batch_tokens`
(an LM-token budget) never filters long clips — but 30 s = 750 frames OOMs the
DiT (~23 GB). The smoke only fit because tier-1 porjai was short. **Fix:** cap
clip duration in the manifest — target ≤14 s, ref ≤8 s (drops 6.8%; short refs
are also better cloning practice). Peak fell 22.9 GB→18.9 GB.

## Smoke run findings (2026-06-15)

Tier-1-only LoRA (porjai, ~2,940 clips), `conf/voxcpm_smoke.yaml`. Purpose was
mechanics, not quality. What we learned:

- **VRAM:** batch_size 2 OOMs on the 24 GB 3090 (peak 22.9 GB). batch_size 1
  peaks at **20.4 GB** — adopted for both smoke and `conf/voxcpm_lora.yaml`.
  Launch training with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Trainable params:** 36.2M / 2326.2M (1.56%) at r=64 on LM+DiT.
- **Generation-length collapse from over-training:** at step 500 the in-training
  snapshots have healthy durations (3.4–5.1 s); by step 1000 every prompt
  collapsed to exactly 1.0 s (stop-token overfit on the tiny single-speaker
  set). → For the real run: fewer epochs (now 2), more/diverse data, and treat
  snapshot duration as a health signal. Track it in Phase 1.
- **Adapter checkpoints load cleanly** into `voxcpm.VoxCPM(lora_weights_path=...)`.

## Reference points (other systems, not directly comparable — different eval sets/ASR)

- JaiTTS-v1.0 (closed): 1.94% short / 2.55% long — beats human GT (1.98% / 2.47%)
- OmniVoice base: 7.71% (FLEURS) — worse than its GT 6.98%

## Caveats on the baseline

- Only 5 short prompts — expand `eval/prompts_*.tsv` before treating any CER as
  precise. The number is a directional "before", good for measuring LoRA deltas.
- Base model run with `optimize=True` (torch.compile) and default cfg=2.5,
  timesteps=10.
