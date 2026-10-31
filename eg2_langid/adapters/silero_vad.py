"""silero VAD 어댑터 (선택) — 음악/효과음 검출 정확도를 높인다.

모델은 패키지에 **포함**되어 있습니다(MIT, Silero Team): eg2_langid/assets/silero_vad.onnx
탐색 순서
  1) EG2_VAD_MODEL 환경변수
  2) 패키지 내장 모델 (기본)
  3) faster-whisper 패키지의 assets/silero_vad*.onnx (설치돼 있으면)
onnxruntime 이 없으면 None 을 반환하며, 호출측은 numpy 특징 경로로 자동 폴백합니다.

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
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ('silero_vad.onnx', 'silero_vad_v6.onnx'):
        cand = os.path.join(here, 'assets', name)
        if os.path.exists(cand):
            return cand
    try:
        import faster_whisper  # noqa: F401
        hits = glob.glob(os.path.join(os.path.dirname(faster_whisper.__file__),
                                      'assets', 'silero_vad*.onnx'))
        if hits:
            return sorted(hits)[0]
    except Exception:
        pass
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
    if {'input', 'state', 'sr'} <= names:      # 저장소 내장 모델(state 결합형)
        layout = 'state'
    elif {'input', 'h', 'c'} <= names:         # v6 (h/c 분리형)
        layout = 'hc'
    else:
        return None

    def vad(pcm) -> float:
        a = np.asarray(pcm, dtype=np.float32).ravel()
        if len(a) < WIN:
            return 0.0
        state = np.zeros((2, 1, 128), np.float32)
        h = np.zeros((1, 1, 128), np.float32)
        c = np.zeros((1, 1, 128), np.float32)
        ctx = np.zeros(CTX, np.float32)
        hits = []
        for st in range(0, len(a) - WIN + 1, WIN):
            x = np.concatenate([ctx, a[st:st + WIN]])[None, :]
            if layout == 'state':
                out = sess.run(None, {'input': x, 'state': state,
                                      'sr': np.array(SR, np.int64)})
                state = out[1]
            else:
                out = sess.run(None, {'input': x, 'h': h, 'c': c})
                h, c = out[1], out[2]
            hits.append(float(np.ravel(out[0])[0]) > threshold)
            ctx = x[0, -CTX:]
        return float(np.mean(hits)) if hits else 0.0

    return vad
