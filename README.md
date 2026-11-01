# eg2-langid

**English** | [한국어](README.ko.md)

Lightweight **language identification from audio**, built on EmbeddingGemma 2 audio embeddings.

A 187 KB linear classifier on top of a 768-dim audio embedding space — **29 languages**,
no LLM calls, CPU-only. The embedding model is **not** bundled: the package attaches to an
existing EmbeddingGemma 2 ONNX runtime and consumes either raw audio or precomputed vectors.

## Features

- **29 languages** (22 with held-out ground truth + 7 added from VoxLingua107)
- **Single 175 KB classifier** — `numpy` is the only hard dependency (a 2.2 MB MIT-licensed silero VAD ships alongside for music/sound-effect detection)
- **Three integration levels**: vector-in · audio + embedder callback · optional model adapter
- **Unknown-language gate** — rejects audio that matches no known language
- **Live desktop widget** — shows the language of whatever is playing on the machine (5 s window)
- **Streaming by design** — embeddings are computed, judged, and discarded; nothing is written to disk
- **One supported build** — the classifier is fitted on **q8-al4** (int8 block, `accuracy_level=4`) embeddings; other quantizations run but shift verdicts in close language pairs (see [Model & quantization](#model--quantization))

## Measured results

| Evaluation | Result |
|---|---|
| Held-out gold set (FLEURS test, 22 languages × 100) | **0.9868** (2,171 / 2,200) |
| Accuracy excluding the id/ms pair | **99.95%** (1 error in 2,000) |
| 5-second window vs full clip | **98.2%** vs 97.7% |
| 3 s / 2 s / 1 s window | 90.5% / 77.3% / 47.0% |
| Unknown-language rejection @ 5.5% false rejection | 37.5% |
| Embedding + judge latency (5 s audio) | ~0.4 s (CPU, 4 threads) |

The remaining errors are concentrated in one pair: Malay ↔ Indonesian (24 + 4 of 29 errors).
These two languages are mutually intelligible to a large degree, so a fraction of read speech
is genuinely ambiguous from audio alone.

## Install

```bash
pip install numpy            # core package — that's all
pip install pyaudiowpatch    # only for the live desktop widget
```

The embedding model is required at runtime but **is not distributed here**.
Point the adapter at an existing deployment directory that contains `mmal4/`, `model-src/` and `src/`:

```bash
export EG2_MMAL4_DIR=/path/to/deploy
```

## Usage

### 1. From a vector (lightest)

```python
from eg2_langid import LanguageIdentifier

li = LanguageIdentifier()
r = li.detect(vec_768d)
print(r)                 # "ko_kr (97%)"
print(r.lang, r.unknown, r.confidence)
```

### 2. From audio with your own embedder

```python
r = li.detect_audio(pcm_float32_16k, embed_fn=my_embedder, window_sec=5.0, hop_sec=1.0)
print(r.distribution)    # {'ko_kr': 0.8, 'en_us': 0.2}  — per-window vote share
```

### 3. With the bundled adapter (needs the model)

```python
from eg2_langid.adapters.gemma2 import make_embedder

embed = make_embedder()                                   # reads EG2_MMAL4_DIR
r = li.detect_audio(pcm_float32_16k, embed_fn=embed, window_sec=5.0)
```

### Result fields

| Field | Meaning |
|---|---|
| `lang` / `label` | winning class; `label` has the {hi, ur} group mapping applied. `None` when rejected |
| `confidence` | **headroom above the unknown-gate threshold** — `0` means barely accepted, `1` means far inside. Not an accuracy figure |
| `p_correct` | **calibrated probability that the verdict is right** (from gold margins) — this is what the widget shows |
| `margin` | top-1 minus top-2 probe score |
| `unknown` | the gate rejected the clip as outside the 29 languages |
| `distribution` | per-window vote share, when the audio is longer than one window |

A short clip can score `confidence = 0.02` and still be correct, so display `p_correct`.

### Music / sound-effect detection

A 5-second window is also scored for **speech content**. When nothing is being spoken the result
carries `kind='music'` or `kind='sfx'` instead of a language, and the widget shows *Music* / *Sound effect*
instead of a (wrong) language.

```python
r = li.detect_audio(pcm, embed_fn=embed, vad_fn=vad)   # vad is optional
r.kind          # 'speech' | 'music' | 'sfx'
r.p_nonspeech   # 0..1

# opt-in: also use the language probe's reaction (17 features instead of 11)
r = li.detect_audio(pcm, embed_fn=embed, vad_fn=vad, use_hybrid=True)
```

`use_hybrid=False` (default) judges music from the **waveform alone** — a music window never pays the
~0.4 s embedding. `use_hybrid=True` adds the probe reaction (top score, margin, entropy, centroid
gap), which measures slightly better out-of-fold (AUC 0.9851 → 0.9918, instrumental 97.5 → 100 %,
YouTube false positives 4.3 → 3.6 %) but requires the embedding on every window. The desktop widget
enables it with `EG2_HYBRID=1`.

| Setting | Speech false-positives | Instrumental | Vocal music | Sound effects |
|---|---|---|---|---|
| **with VAD** (score 0.70) | **0.0 %** | 97.5 % | 75.0 % | 93.6 % |
| without VAD, numpy only (0.85) | 0.0 % | 71.6 % | 68.3 % | 53.2 % |

File-level cross-validation: 129 speech crops (8 languages, FLEURS), 22 instrumental and 20 vocal
tracks, 70 Windows sound effects. `vad` comes from `eg2_langid.adapters.silero_vad.make_vad()`
(the model is **bundled** in `eg2_langid/assets/silero_vad.onnx`, MIT © Silero Team; needs
`onnxruntime` — without it the numpy-only feature path still runs, at lower recall).

**Noisy-speech bias — fixed after a field report.** In a loud club interview or an FPS stream the
first version called too much of it music. Reproduced by mixing speech with music/effects at negative
SNR (music louder than the voice):

| speech mixed with | old model | current model |
|---|---|---|
| music, 0 dB | 6.2 % called music | **0.0 %** |
| music, −5 dB | 18.8 % | **0.0 %** |
| music, −10 dB | **37.5 %** | **0.0 %** |
| game effects, −10 dB | 0.0 % | 0.0 % |
| real YouTube track (80 windows) | 3.8 % | **2.5 %** |
| music/SFX still detected | 95.0 % | 81.7 % |

Two changes produced this: the speech class was retrained with 240 mixed samples (speech + music and
speech + effects at 0 and −5 dB, song/file separated from the test set), and a **VAD guardrail** was
added — if at least 20 % of the window is detected as speech, the music verdict is cancelled. The
guardrail is the bigger lever; the price is that less music is flagged (95 % → 82 %) and vocal tracks
may now show their lyrics' language instead of *Music*.

**Verified on the two videos that exposed the bias** (5 minutes each, 60 windows):

| | club interview (music under the voices) | game stream (SFX + music) |
|---|---|---|
| before | 23.3 % of windows called music | 48.3 % |
| `verify_music` | **3.3 %** | 45.0 % |
| with `EG2_HOLD=20` | **100 % of the time shows a language** | 90.0 % |

The club case is real speech the VAD missed: on those windows the language probe still reported
`en_us` with a margin of +0.59 (85 % of them above 0.4). So a music verdict now triggers one probe
re-check (`verify_music`), rate-limited by `EG2_VERIFY` so music-only playback costs ~7 % CPU instead
of a full embedding per window.

The game stream is different — in the remaining 45 % of windows the probe has no language signal at
all (margin +0.11, none above 0.4); that audio really is SFX only. The hold timer is what keeps the
widget reading "Korean" through those stretches (12 s → 83 %, 20 s → 90 %, 30 s → 98 %).

**Limits, measured**
- Singing is not reliably separated from speech — 75 % of vocal tracks are flagged, the rest show a
  language (which is arguably correct: the lyrics *are* that language).
- Music and sound effects are confused in ~20 % of cases (effect recall 78.7 %), so an alarm may be
  labelled *Music*.
- Trained on 112 source files; a broadcast jingle with heavy speech may still read as a language.

**The head is separate from the language probe** — it scores the raw waveform (VAD ratio, 2–8 Hz
envelope modulation, spectral centroid/flatness/flux, RMS dynamics) with a 11-number logistic
regression, so it cannot degrade the 29-language accuracy (gold stays 0.9868).

#### Why the detector uses signal features, not the embedding

It is tempting to reuse the 768-d embedding — it is already computed, so a music head would be free.
Measured, that choice is dangerous:

| Detector trained on gold speech + 112 music/SFX files | Gold holdout | **Real YouTube speech (80 windows)** |
|---|---|---|
| EG2 embedding, 768-d logistic | AUC **1.000**, detects 100 % | **56.6 % wrongly called music** |
| Language-probe response (margin, entropy, centroid gap) | AUC 0.983, detects 97.7 % | 9.4 % wrongly called music |
| Signal features + VAD (shipped) | AUC 0.984, detects 98.8 % | **3.8 % wrongly called music** |

The embedding head scores perfectly in-distribution because what it learns is *studio recording vs
everything else* — a domain shortcut, not "speech vs music". On real YouTube audio (compression,
reverb, background music) it misfires on more than half the speech. The signal features measure the
thing itself — VAD speech presence and 2–8 Hz syllabic modulation — and hold up.

**Hybrid (signal + probe reaction) — measured, not shipped.** Concatenating the 11 signal features
with the 6 probe-response features (top score, margin, entropy, centroid/cosine gaps) into one
17-feature logistic does improve things, but only marginally:

| 5-fold out-of-fold · threshold at 0.5 % speech FP | AUC | instrumental | vocal | SFX | YouTube FP |
|---|---|---|---|---|---|
| signal only (11, with VAD) | 0.9851 | 97.5 % | 80.0 % | 95.7 % | 4.3 % |
| probe reaction only (6) | 0.9767 | 67.9 % | 18.3 % | 72.3 % | 5.0 % |
| **hybrid (17, C=1)** | **0.9918** | **100 %** | 80.0 % | 97.9 % | **3.6 %** |

It is not used because the probe reaction requires the **embedding**, so every music window would pay
the ~0.4 s embedding cost that the current detector avoids by deciding on the waveform alone. A 0.7 pt
AUC gain does not buy that back on a widget that idles at 0 % CPU.

Reproduce: `data/raw/speech_music/compare_embed_signal.json`, and a 699-second YouTube track gives
3.8 % false positives with median VAD ratio 0.93.

### Live desktop widget

Captures system audio (WASAPI loopback), keeps a 5-second ring buffer and re-judges every 2 s.

```bash
python app/live_langid.py
```

Drag anywhere to move · click **×** to close · right-click for pause / quit.

Add `--serve 8730` to open the HTTP API **inside the same process** (one loaded model instead of two).

**Nothing playing → no compute.** The widget embeds only when new audio has actually arrived, so a
silent machine costs **0 % CPU** (measured). While something is playing it uses **≈1 core**
(2 s hop on a 4-core box; measured 99 %, down from 210 % at a 1 s hop). The window keeps the last
verdict on screen instead of churning through the same frozen buffer.

| Env var | Default | Meaning |
|---|---|---|
| `EG2_HOP` | `2.0` | seconds between judgements (the main CPU lever) |
| `EG2_IDLE_HOP` | `4.0` | slower poll while nothing is playing |
| `EG2_THREADS` | `2` | ONNX Runtime threads for the embedding session |
| `EG2_DEBUG` | — | print `rms` / skip reason per tick |
| `EG2_LANG` | OS locale | widget language: `en` or `ko` |
| `EG2_HOLD` | `20` | seconds a language stays on screen when the audio turns to music |
| `EG2_MODE` | `balanced` | `balanced` · `fast` (3 s window, 1.5 s hop) · `precise` |
| `EG2_WINDOW` | `5` | judging window in seconds — the floor of the response delay |
| `EG2_PREVIEW` | `3` | on speech onset, show a short-window verdict first, marked "checking" (0 disables) |
| `EG2_VERIFY` | `6` | first probe re-check interval after music appears (0 disables) |
| `EG2_VERIFY_MAX` | `20` | back-off ceiling — the interval doubles while music keeps playing |
| `EG2_VERIFY_MARGIN` | `0.4` | probe margin needed to overrule the music verdict |
| `EG2_HYBRID` | `0` | `1` = also use the probe reaction for music detection (slower) |

**CPU cost of the re-check, measured** (10-second samples of the running widget):

| state | music detection OFF (`EG2_NOMUSIC=1`) | music detection ON (default) |
|---|---|---|
| silence | 0 % | **0 %** |
| speech | 121–136 % | **125–150 %** |
| playing music | **117–142 %** | **11–32 %** |

4 × 10 s samples each, 120 s assets. Two things this shows:

- **Music detection is a CPU saver, not a cost.** With it off, music goes through the language probe, so
  every window pays an embedding — music playback then costs as much as speech (≈1.3 cores). With it on,
  music windows are decided from the waveform in ~10 ms and only an occasional probe re-check runs.
- **On speech it costs 5–10 %** (the feature extraction + VAD per window), which is the price of knowing
  when *not* to embed.

Speech is the expensive state either way: judging every 2 s means an embedding every 2 s. Raise
`EG2_HOP` to 3–4 s if that matters more than refresh rate.

The re-check costs one embedding, so running it every 6 s made music playback as expensive as speech.
It now runs immediately when music appears, then backs off 6 → 12 → 20 s while music continues, and
resets to 6 s the moment speech is found. Verified in the debug log: `[verify] delay=6 → 12 → 20`,
then `→ speech` resets it.

**Holding the last language.** An interview over loud background music, or a game stream where the
host pauses to talk to chat, contains windows that really are music-only. Instead of flipping to
*Music* the moment speech stops, the widget keeps the last language visible for `EG2_HOLD` seconds
with a dimmed dot and `music section · holding Korean` / `음악 구간 · 최근 언어 유지`. Only after that
does it show *Music*.

The widget is **English by default** and switches to **Korean** when the Windows UI language is
Korean (`GetUserDefaultUILanguage()` = 0x0412). Language names and every status line follow that
choice; override it with `EG2_LANG=ko` or `EG2_LANG=en`. The percentage shown is `p_correct`.

## Analysing a whole file

The live widget answers *what is playing now*. This answers *what language is spoken where* in a
recording — segments with start/end times plus overall proportions.

```bash
python -m eg2_langid.analyze interview.mp4 --json timeline.json
```

```
file: mix_ko_music_en.wav  0:37 (window 5s · hop 1s)

    0:00–0:14     Korean           99%  (12 windows)
    0:14–0:23     [music/sfx]
    0:23–0:34     English          99%  (11 windows)
    0:34–0:37     [silence]

proportions (speech only): Korean 56.0% · English 44.0%
unknown 0:00 (0.0%) · music 0:09 (24.3%) · silence 0:03 (8.1%)
```

File mode is not in a hurry, so it walks the audio with a **1 s hop** and merges windows into
segments (3-window median smoothing, short runs absorbed).

**Music handling is a policy choice** (`--music`), because excluding music badly is worse than not
excluding it. Measured on a file with known truth (Korean 12 s → music 10 s → English 12 s → silence):

| `--music` | behaviour | result on that file |
|---|---|---|
| **`mark`** (default) | music probability is required *and* the language margin must be low (< `--music-margin 0.3`) | ko 0–14 · **[music] 14–23** · en 23–34 · [silence] 34–37 |
| `off` | no music judgement at all | ko **0–15** · en **15–34** — the music section is counted as language |
| `gate` | music probability alone decides | a speaker over loud music can be swallowed |

The margin rule is what makes `mark` safe: on the same file, music-only windows score margin
0.02–0.11 while speech scores 0.62–0.85, so an interview over loud background music stays speech even
when VAD drops below 0.2 — the exact failure mode that made music detection steal speech in the live
widget.

```bash
python -m eg2_langid.analyze FILE --hop 1 --window 5 --min-seg 2 --music mark --json out.json
```

## Real-time API

The widget is just one frontend — the same engine can feed other programs.

```bash
python -m eg2_langid.server                    # http://127.0.0.1:8730
python -m eg2_langid.server --json             # also NDJSON on stdout
python app/live_langid.py --json               # widget + NDJSON
python -m eg2_langid.server --no-http --json   # pipe only, no ports
```

| endpoint | returns |
|---|---|
| `GET /health` | `{ok, ready, device, window_sec}` |
| `GET /now` | the latest verdict (one JSON object) |
| `GET /events` | NDJSON stream — one line per verdict, keep-alive |
| `POST /detect?window=5` | request body = audio bytes (wav/mp3/m4a/…) → one verdict |

```json
{"seq":42,"ts":1767225600.12,"status":"speech","lang":"ko_kr","label":"ko_kr",
 "p_correct":0.99,"confidence":0.512,"margin":0.6307,"kind":"speech",
 "window_sec":5.0,"process_ms":969,"preview":false,"silent_run":0,
 "device":"스피커(Realtek(R) Audio) [Loopback]"}
```

`status` is one of `starting` · `silent` · `speech` · `unknown` · `music` · `sfx` · `paused` · `error`;
`preview: true` marks the short-window verdict that the next full window replaces. The server binds
`127.0.0.1` by default — pass `--host` to expose it (e.g. a Tailscale address).

## Memory footprint

The model dominates: `q8-al4` weights are **654 MB** (audio encoder 340 MB + main 314 MB). Everything
else is small. Measured in a fresh process:

| stage | RSS |
|---|---|
| Python + numpy + onnxruntime | 97 MB |
| ONNX weights loaded | **732 MB** |
| + first embedding (activations) | 800 MB |
| while judging (5 s window) | 817 MB |
| idle after Windows trims pages | 150–700 MB |

Two things keep it from doubling:

- **One process for widget + API.** `python app/live_langid.py --serve 8730` opens the HTTP API inside
  the widget process, sharing the single loaded model. Running the widget and `eg2_langid.server`
  separately loads the model twice (~2 GB → ~0.8 GB).
- **No torch by default.** The audio frontend (log-mel) is implemented in numpy in
  `adapters/gemma2_np.py`. Measured against the torch reference: log-mel max difference **0.0**,
  embedding cosine **0.99999988**, identical verdicts 24/24, same speed (the ONNX session dominates).
  Skipping torch saves **~240 MB** and drops startup from ~4 s to ~1.3 s. Set `EG2_FRONTEND=torch` to
  use the reference implementation.

### q4-al4 instead of q8-al4 (measured)

The upstream repo publishes a 4-bit build; after injecting `accuracy_level=4` it is a drop-in swap.
Measured on 100 clips (10 languages × 10, 5 s each) against the shipped q8-al4:

| | q8-al4 (default) | q4-al4 |
|---|---|---|
| weights on disk | 654 MB | **363 MB** |
| RSS after load | 731 MB | **509 MB** |
| RSS while judging | 855 MB | **665 MB** |
| embedding cost | 580 ms | 609 ms (**+5 %**) |
| verdicts identical | — | **93 / 100** |
| correct | 92 / 100 | 90 / 100 |

So q4 buys ~190 MB of RSS for ~2 points of accuracy and 5 % speed. Default stays q8; to try q4:

```bash
export EG2_AUDIO_ONNX=/path/audio_encoder_q4_al4.onnx
export EG2_MAIN_ONNX=/path/model_q4_al4.onnx
```

Building those two files: download `onnx/audio_encoder_q4.onnx` and `onnx/model_q4.onnx` (+`_data`)
from `onnx-community/embeddinggemma-2-ONNX`, then set `accuracy_level=4` on every `MatMulNBits` node.

## Supported languages (29)

| Code | Language | Code | Language |
|---|---|---|---|
| `ko_kr` | Korean | `it_it` | Italian |
| `en_us` | English | `nl_nl` | Dutch |
| `ja_jp` | Japanese | `pl_pl` | Polish |
| `cmn_hans_cn` | Chinese (Mandarin, simplified) | `ru_ru` | Russian |
| `es_419` | Spanish (Latin America) | `uk_ua` | Ukrainian |
| `pt_br` | Portuguese (Brazil) | `sv_se` | Swedish |
| `fr_fr` | French | `ar_eg` | Arabic |
| `de_de` | German | `hi_in` | Hindi *(also reports Urdu)* |
| `th_th` | Thai | `id_id` | Indonesian |
| `tr_tr` | Turkish | `ms_my` | Malay |
| `vi_vn` | Vietnamese | `fil_ph` | Filipino |
| `ur_pk` | Urdu | `da_dk` | Danish |
| `fa_ir` | Persian | `cs_cz` | Czech |
| `hr_hr` | Croatian | `hy_am` | Armenian |
| `sw_ke` | Swahili | | |

Two structural notes:

- **Urdu is reported as Hindi** (`hi_in/ur_pk` group). The pair does not separate reliably on
  read speech, so the group label is returned instead of a hard 50/50 guess.
- **Indonesian and Malay** get a second-stage pairwise decision; the pair still tops out around
  96% / 76% because of mutual intelligibility.

## Repository layout

```
eg2_langid/          distributable package (numpy only, no model weights)
  detector.py        probe + group + pairwise head + unknown gate
  capture.py         WASAPI loopback ring buffer (reopen + resample)
  engine.py          LiveEngine — boot, judge loop, State for UI/API
  analyze.py         whole-file timeline + proportions (CLI)
  server.py          HTTP (/now /events /detect) + NDJSON
  nonspeech.py       music / sfx detector (11 signal features + VAD guardrail)
  adapters/gemma2.py optional embedding adapter (onnxruntime / torch)
  adapters/silero_vad.py  optional VAD adapter
  data/              model.npz (175 KB) + labels.json
  assets/            silero_vad.onnx (2.2 MB, MIT) + its LICENSE
app/live_langid.py   live desktop widget (Tk, thin frontend over LiveEngine)
tools/               artifact export + reference embedder
tests/               self-tests: package, engine, analyze, server, e2e
examples/            quickstart
```

`tools/export_artifact.py` regenerates `model.npz` from a local dataset directory and reproduces
the reported gold accuracy.

## Model & quantization

The probe consumes 768-dim L2-normalized audio embeddings. The embedding model is not bundled —
any EmbeddingGemma 2 ONNX build that exposes the sentence-embedding output works.

**What the numbers in this README were measured with**

| Item | Value |
|---|---|
| Weight format | **int8 block quantization — `MatMulNBits` with `bits=8`, `accuracy_level=4`** ("q8-al4") |
| Files | audio encoder 340 MB + backbone 314 MB (**654 MB**) |
| Resident memory | ≈ 0.6 GB |
| Latency | 0.4–1.2 s per 27 s chunk (CPU, 4 threads) |

Node-level check of the shipped build: 123 `MatMulNBits` (bits=8) in the audio encoder and 146 in
the backbone — i.e. **int8 weights**, matching the `*_quantized.onnx` variants published upstream.

**Why q8 on CPU**

| Variant | Disk | CPU memory | Notes |
|---|---|---|---|
| **q8 (int8 block)** | 654 MB | **lowest of the quantized set** | the build behind every figure here |
| q4 (int4 block) | **363 MB** | **lower** | measured: RSS 855 → 665 MB, **5 % slower**, verdicts 93/100 identical, accuracy 92 → 90 per 100 |
| q4f16 | 327 MB | **higher** | keeps fp16 activations alive — measured ≈559 MB commit vs ≈466 MB for q8 on this model family |
| fp16 | 1.1 GB | highest | reference quality |

On CPU, **q8 is the sweet spot**: int8 weights with int8 compute (`accuracy_level=4`) keep the
working set smaller than q4f16 despite the larger file, because activations stay in a compact form.
q4 only wins on disk size.

**The shipped probe is trained on q8-al4 embeddings — keep the runtime matched.** `model.npz` was
fitted on q8-al4 vectors, so **q8-al4 is the supported configuration**. Another build still runs (the
probe is linear, so it is not tied to a quantizer), but the embedding drift moves decisions exactly
where the margins are thin:

| build vs q8-al4 | embedding cosine | margin \|Δ\| (median / p90) | verdicts changed | accuracy |
|---|---|---|---|---|
| q4 audio encoder only | 0.997 | 0.011 / 0.033 | 2 / 100 | 92 → 92 |
| q4 backbone only | 0.987 | 0.037 / 0.083 | 3 / 100 | 92 → 89 |
| all q4 | 0.984 | 0.043 / 0.088 | 4 / 100 | 92 → 90 |

Every changed verdict landed in a **close pair** — Indonesian/Malay (margin ≈ 0.06) or
Russian/Ukrainian (≈ 0.31) — where the drift is the same size as the margin. The other 27 languages
have margins 0.5–0.75, i.e. **10–30× the drift**, and do not move.

If you must ship q4: **retrain `model.npz` on q4 vectors** (`tools/export_artifact.py`, same
procedure — re-embedding the 9,880 training vectors takes ≈1.6 h single-process) and never mix vectors
from two builds in one index. The same clip already differs by cosine 0.997 between builds, which is
enough to corrupt similarity search and duplicate detection.

**Do not run quantized weights on a GPU execution provider.** `MatMulNBits` has no DirectML or CUDA
kernel — those nodes silently fall back to CPU, so you reserve GPU memory without getting GPU speed.

> Everything reported here was measured with the **q8-al4** build, and the bundled probe expects it.
> The upstream repo also publishes fp32 / fp16 / q4 / q4f16 builds — the adapter accepts any of them as
> a file swap (`EG2_AUDIO_ONNX` / `EG2_MAIN_ONNX`), but read the table above before switching: the build
> and the probe have to match, and quantized variants cannot be mixed inside one vector index.

## Limitations

- Windows is the only platform tested for the live widget (WASAPI loopback); the package itself is portable.
- Windows capture runs at the device rate (e.g. 192 kHz) and is resampled to 16 kHz in memory.
- Windows shorter than ~3 s lose accuracy quickly; 5 s is recommended.
- Music and non-speech are handled by the gate ("unknown"), not by a dedicated class.
- Malay/Indonesian ambiguity is intrinsic and cannot be fully resolved with more data.

## License

This repository (code and the trained probe artifact) is released under the **MIT License** — see [LICENSE](LICENSE).

### Third-party attribution

The probe artifact was trained on embeddings of two public speech datasets. Neither dataset's
audio is redistributed in this repository; the attribution below is provided in accordance with
their licenses.

- **silero-vad** (Silero Team) — **MIT** — `eg2_langid/assets/silero_vad.onnx`, redistributed here
  unchanged (2.2 MB). License text: `eg2_langid/assets/SILERO_VAD_LICENSE.txt`.
  Source: <https://github.com/snakers4/silero-vad> (commit-pinned copy, sha256 in `assets/README.md`).
- **FLEURS** (Google) — CC BY 4.0 — used for training and evaluation data.
- **VoxLingua107** (TalTech) — CC BY 4.0 — used for additional language coverage.
- **EmbeddingGemma 2** model weights are **not** included in this repository and are governed by
  their own license (Gemma Terms of Use). Obtain them separately.
