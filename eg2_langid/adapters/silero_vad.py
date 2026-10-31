"""silero VAD 어댑터 (선택) — 음악/효과음 검출 정확도를 높인다.

onnxruntime 이 필요하며, 모델 파일은 다음 순서로 찾습니다.
  1) EG2_VAD_MODEL 환경변수
  2) faster-whisper 패키지의 assets/silero_vad_v6.onnx (설치돼 있으면)

사용:
    from eg2_langid.adapters.silero_vad import make_vad
    vad = make_vad()                 # 없으면 None
    r = li.detect_audio(pcm, embed_fn=embed, vad_fn=vad)
"""
from __future__ import annotations

import glob
import os

import numpy as np

SR = 16000
CTX, WIN = 64, 512          # silero v6: 입력 576 = 컨텍스트 64 + 창 512


def find_model() -> str | None:
    p = os.environ.get('EG2_VAD_MODEL')
    if p and os.path.exists(p):
        return p
    pats = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(__file__)))), '**', 'silero_vad*.onnx'),
    ]
    try:
        import faster_whisper  # noqa: F401
        base = os.path.dirname(faster_whisper.__file__)
        pats.insert(0, os.path.join(base, 'assets', 'silero_vad*.onnx'))
    except Exception:
        pass
    for pat in pats:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    return None


def make_vad(model_path: str | None = None, threshold: float = 0.5):
    """발화 비율 함수(pcm → 0~1)를 반환. 사용 불가면 None."""
    path = model_path or find_model()
    if not path:
        return None
    try:
        import onnxruntime as ort
    except Exception:
        return None
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    names = {i.name for i in sess.get_inputs()}
    if names != {'input', 'h', 'c'}:
        return None

    def vad(pcm) -> float:
        a = np.asarray(pcm, dtype=np.float32).ravel()
        if len(a) < WIN:
            return 0.0
        h = np.zeros((1, 1, 128), np.float32)
        c = np.zeros((1, 1, 128), np.float32)
        ctx = np.zeros(CTX, np.float32)
        hits = []
        for st in range(0, len(a) - WIN + 1, WIN):
            x = np.concatenate([ctx, a[st:st + WIN]])[None, :]
            out = sess.run(None, {'input': x, 'h': h, 'c': c})
            hits.append(float(out[0].ravel()[0]) > threshold)
            h, c = out[1], out[2]
            ctx = x[0, -CTX:]
        return float(np.mean(hits)) if hits else 0.0

    return vad
