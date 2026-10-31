# eg2-langid

**English** | [한국어](README.ko.md)

Lightweight **language identification from audio**, built on EmbeddingGemma 2 audio embeddings.

A 187 KB linear classifier on top of a 768-dim audio embedding space — **29 languages**,
no LLM calls, CPU-only. The embedding model is **not** bundled: the package attaches to an
existing EmbeddingGemma 2 ONNX runtime and consumes either raw audio or precomputed vectors.

## Features

- **29 languages** (22 with held-out ground truth + 7 added from VoxLingua107)
- **Single 187 KB artifact** — `numpy` is the only hard dependency
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

### Live desktop widget

Captures system audio (WASAPI loopback), keeps a 5-second ring buffer and updates once per second.

```bash
python app/live_langid.py
```

Drag to move · right-click to pause or quit.

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
  data/              model.npz (187 KB) + labels.json
app/live_langid.py   live desktop widget (WASAPI loopback)
tools/               artifact export + reference embedder
tests/               package self-test and end-to-end check
examples/            quickstart
```

`tools/export_artifact.py` regenerates `model.npz` from a local dataset directory and reproduces
the reported gold accuracy.

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

- **FLEURS** (Google) — CC BY 4.0 — used for training and evaluation data.
- **VoxLingua107** (TalTech) — CC BY 4.0 — used for additional language coverage.
- **EmbeddingGemma 2** model weights are **not** included in this repository and are governed by
  their own license (Gemma Terms of Use). Obtain them separately.
