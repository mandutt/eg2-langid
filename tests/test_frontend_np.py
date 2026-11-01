"""numpy 전처리(gemma2_np)가 torch 참조 구현과 동일한지 검사.

torch/open_eg 또는 FLEURS 캐시가 없으면 SKIP.
실행: python tests/test_frontend_np.py
"""
from __future__ import annotations

import io
import os
import sys
import tarfile

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'data', 'raw', 'deploy', 'src'))

FAIL = []
CACHE = os.path.join(BASE, 'data', 'raw', 'fleurs_cache')
SR = 16000


def check(name, cond, detail=''):
    print(('PASS  ' if cond else 'FAIL  ') + name + (f'  ({detail})' if detail else ''))
    if not cond:
        FAIL.append(name)


def clips(lang: str, n: int, secs: float = 5.0):
    import soundfile as sf
    path = os.path.join(CACHE, f'{lang}_test.tar.gz')
    if not os.path.exists(path):
        return []
    out = []
    with tarfile.open(path) as tf:
        for m in tf.getmembers():
            if not m.name.endswith('.wav'):
                continue
            a, sr = sf.read(io.BytesIO(tf.extractfile(m).read()), dtype='float32')
            if a.ndim > 1:
                a = a.mean(axis=1)
            if sr != SR or len(a) < secs * SR:
                continue
            out.append(a[:int(secs * SR)])
            if len(out) >= n:
                break
    return out


def main():
    from eg2_langid.adapters.gemma2_np import log_mel, make_embedder as make_np

    wavs = [w for l in ('ko_kr', 'en_us') for w in clips(l, 3)]
    if not wavs:
        print('SKIP  FLEURS 캐시 없음')
        return 0

    # ① mel 단계 비교 (torch 있을 때만)
    try:
        import torch                                              # noqa: F401
        from open_eg.processor import preprocess_audio
        f_t, _ = preprocess_audio([torch.from_numpy(wavs[0])])
        d = float(np.abs(f_t[0].numpy() - log_mel(wavs[0])).max())
        check('log-mel 동일 (최대차 < 1e-6)', d < 1e-6, f'{d:.2e}')
    except Exception as e:
        print(f'SKIP  torch 비교 — {type(e).__name__}')

    # ② 임베딩 비교
    try:
        from eg2_langid.adapters.gemma2 import make_embedder as make_torch
        et = make_torch(threads=2, frontend='torch')
        have_torch = True
    except Exception as e:
        print(f'SKIP  torch 임베더 — {type(e).__name__}')
        have_torch = False

    en = make_np(threads=2)
    vn = [en(w) for w in wavs]
    check('numpy 임베딩 L2=1', all(abs(float(np.linalg.norm(v)) - 1) < 1e-4 for v in vn))
    check('numpy 임베딩 차원', vn[0].shape == (768,), str(vn[0].shape))

    if have_torch:
        vt = [et(w) for w in wavs]
        cos = [float(np.dot(a, b)) for a, b in zip(vt, vn)]
        check('torch↔numpy cosine > 0.9999', min(cos) > 0.9999, f'min {min(cos):.8f}')

        from eg2_langid import LanguageIdentifier
        li = LanguageIdentifier()
        same = sum(li.detect(a).label == li.detect(b).label for a, b in zip(vt, vn))
        check('판정 일치', same == len(wavs), f'{same}/{len(wavs)}')

    # ③ 128 배수가 아닌 길이 — 패딩 경계에서 인덱스를 넘지 않고 참조와 일치해야 한다
    odd = [int(s * SR) for s in (0.5, 1.37, 3.7, 7.03)]
    bad = [n for n in odd if n % 128 == 0]
    check('홀수 길이 준비', not bad, str(bad))
    try:
        vn_odd = [en(wavs[0][:n]) for n in odd]
        check('홀수 길이 임베딩 L2=1', all(abs(float(np.linalg.norm(v)) - 1) < 1e-4 for v in vn_odd))
        if have_torch:
            vt_odd = [et(wavs[0][:n]) for n in odd]
            cos_odd = [float(np.dot(a, b)) for a, b in zip(vt_odd, vn_odd)]
            check('홀수 길이 torch↔numpy cosine > 0.9999', min(cos_odd) > 0.9999,
                  f'min {min(cos_odd):.8f}')
    except IndexError as e:
        check('홀수 길이 인덱스 안전', False, str(e))

    # ④ numpy 경로만 쓰는 별도 프로세스에서 torch가 로드되지 않아야 한다
    import subprocess
    code = ('import sys; sys.path.insert(0, r"%s"); '
            'from eg2_langid.adapters.gemma2_np import make_embedder; '
            'print("torch" in sys.modules)' % BASE)
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                         cwd=BASE).stdout.strip()
    check('별도 프로세스에서 torch 미로드', out == 'False', out or '(출력 없음)')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
