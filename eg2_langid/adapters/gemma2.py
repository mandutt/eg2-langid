"""선택적 어댑터 — 이미 구동 중인 EmbeddingGemma2 ONNX에 붙는 임베딩 함수.

모델은 이 패키지에 포함되지 않는다. 다음 순서로 경로를 찾는다:
  1) 환경변수 EG2_MMAL4_DIR (mmal4/ + model-src/ + src/ 를 담은 디렉터리)
  2) 레포 기본 경로 ../data/raw/deploy
  3) scripts/ (개발 체크아웃)

사용:
    from eg2_langid.adapters.gemma2 import make_embedder
    embed = make_embedder()          # 768d L2 정규화 벡터 반환
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np

_CANDIDATES = [
    os.environ.get('EG2_MMAL4_DIR'),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 'data', 'raw', 'deploy'),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 'scripts'),
]
SR = 16000


def _find_pkg() -> str:
    for c in _CANDIDATES:
        if not c:
            continue
        if os.path.isdir(os.path.join(c, 'mmal4')) and os.path.isdir(os.path.join(c, 'src')):
            return c
    raise FileNotFoundError(
        'EmbeddingGemma2 배포 디렉터리를 찾지 못했습니다. EG2_MMAL4_DIR 를 지정하세요 '
        '(mmal4/ + model-src/ + src/ 포함).')


def make_embedder(pkg_dir: str | None = None, threads: int = 4, max_sec: float = 30.0):
    """embed_fn(pcm_float32_mono_16k) -> 768d float32 (L2 정규화)."""
    d = pkg_dir or _find_pkg()
    src = open(os.path.join(d, 'embed_worker.py'), encoding='utf-8').read()
    if 'if __name__' in src:
        src = src.split('if __name__')[0]
    mod = types.ModuleType('eg2_embedder')
    mod.__file__ = os.path.join(d, 'embed_worker.py')
    exec(compile(src, 'embed_worker.py', 'exec'), mod.__dict__)

    tok, aenc, main_o, empty, pre, bmid, nsoft = mod.build_model(d, threads)
    import torch

    max_n = int(max_sec * SR)

    def embed(pcm: np.ndarray) -> np.ndarray:
        a = np.asarray(pcm, dtype=np.float32).ravel()
        if len(a) > max_n:
            a = a[-max_n:]
        feats, mask = pre([torch.from_numpy(a)])
        af = aenc.run(['audio_features'], {'input_features': feats.numpy(),
                                          'input_features_mask': mask.numpy()})[0]
        ids = bmid([('audio', nsoft(mask[0]))], lambda s: tok.encode(s, add_special_tokens=False))
        v = main_o.run(['sentence_embedding'],
                       {'input_ids': np.array([ids], dtype=np.int64),
                        'attention_mask': np.ones((1, len(ids)), dtype=np.int64),
                        'image_features': empty, 'video_features': empty,
                        'audio_features': af})[0][0].astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-12)

    embed.pcm_hint = SR
    return embed
