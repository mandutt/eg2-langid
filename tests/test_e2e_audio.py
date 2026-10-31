#!/usr/bin/env python
"""엔드투엔드 검증 — FLEURS 클립을 실제로 스피커로 재생하고 루프백으로 받아 판정.

검증 항목:
  1) 루프백 캡처가 실제 소리를 담는가 (RMS > 0)
  2) 캡처된 오디오의 판정이 원 언어와 일치하는가
  3) 직접 계산한 벡터와 루프백 벡터의 코사인 (경로 일관성)
"""
import os, sys, time, tarfile, struct, json
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, 'app'))
from live_langid import LoopbackCapture, SR, DISPLAY   # noqa: E402
from eg2_langid import LanguageIdentifier              # noqa: E402
from eg2_langid.adapters.gemma2 import make_embedder    # noqa: E402
import pyaudiowpatch as pyaudio                         # noqa: E402


def play(a, sr=SR):
    """PyAudio로 스피커 재생 (별도 스레드)."""
    import threading
    def _run():
        pa = pyaudio.PyAudio()
        try:
            s = pa.open(format=pyaudio.paFloat32, channels=1, rate=sr, output=True)
            s.write(a.astype(np.float32).tobytes())
            s.stop_stream(); s.close()
        finally:
            pa.terminate()
    t = threading.Thread(target=_run, daemon=True); t.start()
    return t

CACHE = os.path.join(BASE, 'data', 'raw', 'fleurs_cache')
TARGETS = ['ko_kr', 'en_us', 'ja_jp']


def first_wav(lang):
    tarp = os.path.join(CACHE, f'{lang}_test.tar.gz')
    with tarfile.open(tarp, 'r:gz') as tf:
        for m in tf:
            if not m.name.endswith('.wav'):
                continue
            raw = tf.extractfile(m).read()
            if len(raw) < 16000 * 2 * 6:
                continue
            fmt = struct.unpack('<H', raw[20:22])[0]
            pos, a = 12, None
            while pos < len(raw) - 8:
                csz = struct.unpack('<I', raw[pos + 4:pos + 8])[0]
                if raw[pos:pos + 4] == b'data':
                    dsz = min(csz - (csz % 4), len(raw) - pos - 8)
                    a = (np.frombuffer(raw[pos + 8:pos + 8 + dsz], '<f4') if fmt == 3
                         else np.frombuffer(raw[pos + 8:pos + 8 + dsz], '<i2').astype(np.float32) / 32768.0)
                    break
                pos += 8 + csz + (csz % 2)
            if a is not None:
                return np.asarray(a, np.float32)
    return None


def main():
    li = LanguageIdentifier()
    embed = make_embedder(threads=4)
    cap = LoopbackCapture(); cap.start()
    time.sleep(1.5)
    if cap.error:
        print('캡처 실패', cap.error); return 1
    print(f'장치: {cap.device_name} @ {cap.sr}Hz')

    results = []
    for lang in TARGETS:
        a = first_wav(lang)
        if a is None:
            print(f'{lang}: wav 없음'); continue
        a = a[:int(8.0 * SR)]                      # 8초만
        direct = embed(a)                          # 경로 A: 직접 임베딩
        th = play(a, SR);                          # 스피커 재생
        time.sleep(6.0)                            # 5초 창이 채워질 시간
        buf = cap.resample_to(cap.snapshot(5.0), SR)
        rms = float(np.sqrt(np.mean(buf ** 2)))
        r = li.detect_audio(buf, embed_fn=embed, window_sec=5.0)
        via = embed(buf)
        cos = float(np.dot(direct, via))
        results.append((lang, rms, r, cos))
        print(f'{lang:6} RMS {rms:.4f} · 판정 {r!s:28} · cos(직접,루프백) {cos:.3f}')

    cap.running = False
    ok = sum(1 for lang, rms, r, cos in results if rms > 1e-3 and not r.unknown
             and r.label.split('/')[0] == lang)
    print(f'\n일치 {ok}/{len(results)}')
    return 0 if ok >= max(1, len(results) - 1) else 1


if __name__ == '__main__':
    sys.exit(main())
