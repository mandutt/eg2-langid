#!/usr/bin/env python
"""테스트용 재생기 — FLEURS 클립을 순환 재생 (라이브 창 확인용)."""
import os, sys, time, tarfile, struct
import numpy as np
import pyaudiowpatch as pyaudio

CACHE = r'C:\code\eg2-langidentifier\data\raw\fleurs_cache'
SR = 16000
LANGS = sys.argv[1].split(',') if len(sys.argv) > 1 else ['ko_kr', 'en_us', 'ja_jp']
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 40


def wav(lang):
    with tarfile.open(os.path.join(CACHE, f'{lang}_test.tar.gz'), 'r:gz') as tf:
        for m in tf:
            if not m.name.endswith('.wav'):
                continue
            raw = tf.extractfile(m).read()
            if len(raw) < 16000 * 2 * 6:
                continue
            fmt = struct.unpack('<H', raw[20:22])[0]
            pos = 12
            while pos < len(raw) - 8:
                csz = struct.unpack('<I', raw[pos + 4:pos + 8])[0]
                if raw[pos:pos + 4] == b'data':
                    dsz = min(csz - (csz % 4), len(raw) - pos - 8)
                    a = (np.frombuffer(raw[pos + 8:pos + 8 + dsz], '<f4') if fmt == 3
                         else np.frombuffer(raw[pos + 8:pos + 8 + dsz], '<i2').astype(np.float32) / 32768.0)
                    return np.asarray(a, np.float32)[:8 * SR]
                pos += 8 + csz + (csz % 2)
    return None


pa = pyaudio.PyAudio()
s = pa.open(format=pyaudio.paFloat32, channels=1, rate=SR, output=True)
t0 = time.time()
i = 0
while time.time() - t0 < DUR:
    lang = LANGS[i % len(LANGS)]
    a = wav(lang)
    if a is not None:
        print(f'[{time.time()-t0:5.1f}s] {lang}', flush=True)
        s.write(a.tobytes())
    i += 1
    time.sleep(0.3)
s.stop_stream(); s.close(); pa.terminate()
