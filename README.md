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

The widget is **English by default** and switches to **Korean** when the Windows UI language is
Korean (`GetUserDefaultUILanguage()` = 0x0412). Language names and every status line follow that
choice; override it with `EG2_LANG=ko` or `EG2_LANG=en`. The percentage shown is `p_correct`.

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
  adapters/gemma2.py optional embedding adapter (onnxruntime / torch)
  data/              model.npz (175 KB) + labels.json
  assets/            silero_vad.onnx (2.2 MB, MIT) + its LICENSE
app/live_langid.py   live desktop widget (WASAPI loopback)
tools/               artifact export + reference embedder
tests/               package self-test and end-to-end check
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
| q4 (int4 block) | 363 MB | similar | available upstream (`*_q4.onnx`); smaller on disk, not benchmarked here |
| q4f16 | 327 MB | **higher** | keeps fp16 activations alive — measured ≈559 MB commit vs ≈466 MB for q8 on this model family |
| fp16 | 1.1 GB | highest | reference quality |

On CPU, **q8 is the sweet spot**: int8 weights with int8 compute (`accuracy_level=4`) keep the
working set smaller than q4f16 despite the larger file, because activations stay in a compact form.
q4 only wins on disk size.

**Do not run quantized weights on a GPU execution provider.** `MatMulNBits` has no DirectML or CUDA
kernel — those nodes silently fall back to CPU, so you reserve GPU memory without getting GPU speed.

> Everything reported here was measured with the q8-al4 build. The upstream repo also publishes
> fp32 / fp16 / q4 / q4f16 builds; switching is a file swap, and the adapter is quantization-agnostic.

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
