"""EmbeddingGemma2 오디오 임베딩 — numpy 전처리 + ONNX (torch 불필요).

open_eg.processor 의 log-mel 을 numpy 로 옮긴 것. 오디오 전용 경로에서는 토크나이저도 쓰지 않는다.
참조 구현(torch)과 비트 수준으로 같도록 float64로 계산한 뒤 float32로 내린다.

    from eg2_langid.adapters.gemma2_np import make_embedder
    embed = make_embedder(threads=2)
    v = embed(pcm_float32_mono_16k)     # 768차원, L2 정규화
"""
from __future__ import annotations

import os

import numpy as np

SR = 16000
BOS_TOKEN_ID = 2
EOS_TOKEN_ID = 1
BOA_TOKEN_ID = 256_000
AUDIO_TOKEN_ID = 258_881
EOA_TOKEN_ID = 258_883

FRAME = 320          # 20 ms
HOP = 160            # 10 ms
FFT = 512
N_MELS = 128
MEL_FLOOR = 1e-3

_CANDIDATES = [
    os.environ.get('EG2_MMAL4_DIR'),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 'data', 'raw', 'deploy'),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 'scripts'),
]


def find_pkg(pkg_dir: str | None = None) -> str:
    for c in ([pkg_dir] if pkg_dir else []) + _CANDIDATES:
        if c and os.path.isdir(c) and os.path.isdir(os.path.join(c, 'mmal4')):
            return c
    raise FileNotFoundError(
        'EmbeddingGemma2 배포 디렉터리(mmal4/)를 찾지 못했습니다. EG2_MMAL4_DIR 를 지정하세요.')


def _hann(n: int = FRAME) -> np.ndarray:
    """torch.hann_window(n, periodic=True) 와 동일."""
    k = np.arange(n, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * k / n)


def _mel_bank(n_freqs: int = FFT // 2 + 1, n_mels: int = N_MELS,
              f_min: float = 0.0, f_max: float = SR / 2) -> np.ndarray:
    """HTK 삼각 멜 필터 (면적 정규화 없음) — (n_freqs, n_mels)."""
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    mels = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2, dtype=np.float64)
    edges = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    fft_freqs = np.linspace(0, SR // 2, n_freqs, dtype=np.float64)
    diff = edges[1:] - edges[:-1]
    slopes = edges[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    return np.clip(np.minimum(down, up), 0.0, None)


_WINDOW = _hann()
_MEL = _mel_bank()


def log_mel(wav: np.ndarray) -> np.ndarray:
    """(S,) float32 16 kHz → (frames, 128) float32 log-mel (semicausal, 20 ms/10 ms)."""
    x = np.asarray(wav, np.float32).ravel().astype(np.float64)
    x = np.pad(x, (FRAME // 2, 0))                       # 첫 프레임이 t=0 중심
    n_frames = max(0, (len(x) - (FRAME + 1)) // HOP + 1)
    if n_frames <= 0:
        return np.zeros((0, N_MELS), np.float32)
    idx = np.arange(n_frames)[:, None] * HOP + np.arange(FRAME)[None, :]
    frames = x[idx]
    spec = np.abs(np.fft.rfft(frames * _WINDOW, n=FFT))
    mel = spec @ _MEL
    return np.log(mel + MEL_FLOOR).astype(np.float32)


def frame_mask(n_samples: int, n_frames: int) -> np.ndarray:
    """실제 샘플만 포함한 프레임만 True (패딩 구간 제외)."""
    ends = np.arange(n_frames) * HOP + FRAME
    sm = np.concatenate([np.zeros(FRAME // 2, bool), np.ones(n_samples, bool)])
    return sm[ends]


def make_embedder(pkg_dir: str | None = None, threads: int = 4,
                  max_sec: float = 30.0, audio_onnx: str | None = None,
                  main_onnx: str | None = None):
    """embed_fn(pcm_float32_mono_16k) -> 768d float32 (L2 정규화). torch 없이 동작.

    audio_onnx/main_onnx 로 다른 양자화 빌드(q4 등)를 직접 지정할 수 있다.
    """
    import onnxruntime as ort

    pkg = find_pkg(pkg_dir)
    mm = os.path.join(pkg, 'mmal4')
    # 다른 양자화 빌드(q4 등)는 환경변수로 지정할 수 있다
    audio_onnx = audio_onnx or os.environ.get('EG2_AUDIO_ONNX')
    main_onnx = main_onnx or os.environ.get('EG2_MAIN_ONNX')
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(threads)
    aenc = ort.InferenceSession(audio_onnx or os.path.join(mm, 'audio_encoder_quantized_al4.onnx'),
                                sess_options=so, providers=['CPUExecutionProvider'])
    main_o = ort.InferenceSession(main_onnx or os.path.join(mm, 'model_quantized_al4.onnx'),
                                  sess_options=so, providers=['CPUExecutionProvider'])
    empty = np.zeros((0, 512), np.float32)
    max_n = int(max_sec * SR)

    def embed(pcm: np.ndarray) -> np.ndarray:
        x = np.asarray(pcm, np.float32).ravel()[:max_n]
        n = len(x)
        longest = -(-n // 128) * 128                     # 128 배수로 패딩 (참조 구현과 동일)
        xp = np.pad(x, (0, longest - n)) if longest > n else x
        feats = log_mel(xp)[None]
        fmask = frame_mask(n, feats.shape[1])[None]
        af = aenc.run(['audio_features'],
                      {'input_features': feats, 'input_features_mask': fmask})[0]
        n_soft = int(fmask[0][::2][::2].sum())
        ids = np.array([[BOS_TOKEN_ID, BOA_TOKEN_ID] + [AUDIO_TOKEN_ID] * n_soft
                        + [EOA_TOKEN_ID, EOS_TOKEN_ID]], np.int64)
        v = main_o.run(['sentence_embedding'],
                       {'input_ids': ids,
                        'attention_mask': np.ones((1, ids.shape[1]), np.int64),
                        'image_features': empty, 'video_features': empty,
                        'audio_features': af})[0]
        return (v[0] / (np.linalg.norm(v[0]) + 1e-12)).astype(np.float32)

    return embed
