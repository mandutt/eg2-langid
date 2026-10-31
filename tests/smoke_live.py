#!/usr/bin/env python
"""라이브 파이프라인 스모크 (GUI 없이) — 루프백 캡처 → 임베딩 → 판정."""
import os, sys, time
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'app'))
from live_langid import LoopbackCapture, SR, DISPLAY   # noqa: E402
from eg2_langid import LanguageIdentifier              # noqa: E402
from eg2_langid.adapters.gemma2 import make_embedder    # noqa: E402

print('1) 루프백 장치 확인')
cap = LoopbackCapture()
cap.start()
time.sleep(2.0)
if cap.error:
    print('   캡처 오류:', cap.error)
    sys.exit(1)
print(f'   장치: {cap.device_name} · {cap.sr}Hz')

print('2) 모델 로드')
t0 = time.time()
embed = make_embedder(threads=4)
print(f'   {time.time()-t0:.1f}s')

li = LanguageIdentifier()
print('3) 6초 캡처 후 판정')
time.sleep(max(0, 6.0 - (time.time() - t0)))
a = cap.resample_to(cap.snapshot(5.0), SR)
rms = float(np.sqrt(np.mean(a ** 2))) if len(a) else 0.0
print(f'   버퍼 {len(a)/SR:.1f}초 · RMS {rms:.4f}')
t1 = time.time()
r = li.detect_audio(a, embed_fn=embed, window_sec=5.0)
print(f'   판정 {(time.time()-t1)*1000:.0f}ms → {r}')
print(f'   상위: {list(r.scores.items())[:3]}')
if rms < 1e-4:
    print('   ※ 무음(재생 중인 소리 없음) — unknown이 정상')
cap.running = False
