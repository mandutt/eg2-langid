"""HTTP API 검사 — 서버를 띄워 /health /now /detect /events 를 확인한다.

모델이 없으면 SKIP. 실행: python tests/test_server.py [--port 8791]
"""
from __future__ import annotations

import io
import json
import os
import socket
import struct
import subprocess
import sys
import time
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

FAIL = []


def check(name, cond, detail=''):
    print(('PASS  ' if cond else 'FAIL  ') + name + (f'  ({detail})' if detail else ''))
    if not cond:
        FAIL.append(name)


def free_port() -> int:
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wav_bytes(sec: float = 3.0, sr: int = 16000) -> bytes:
    """간단한 톤 wav — 판정 정확도가 아니라 API 동작만 본다."""
    import numpy as np
    t = np.arange(int(sec * sr)) / sr
    a = (0.2 * np.sin(2 * np.pi * 220 * t)).astype('<f4')
    data = a.tobytes()
    hdr = b'RIFF' + struct.pack('<I', 36 + len(data)) + b'WAVEfmt ' + \
          struct.pack('<IHHIIHH', 16, 1, 1, sr, sr * 2, 2, 16) + b'data' + struct.pack('<I', len(data))
    return hdr + data


def get(url: str, timeout: float = 10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read()


def main():
    port = int(sys.argv[sys.argv.index('--port') + 1]) if '--port' in sys.argv else free_port()
    env = dict(os.environ)
    p = subprocess.Popen([sys.executable, '-u', '-m', 'eg2_langid.server', '--port', str(port)],
                         cwd=BASE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True)
    base = f'http://127.0.0.1:{port}'
    ready = False
    for _ in range(60):
        time.sleep(1)
        try:
            code, body = get(f'{base}/health', timeout=3)
            if code == 200 and json.loads(body).get('ready'):
                ready = True
                break
        except Exception:
            if p.poll() is not None:
                break
    if not ready:
        out = p.stdout.read() if p.stdout else ''
        print('SKIP  서버 준비 실패 (모델 없음?)', out.strip()[-200:])
        p.terminate()
        return 0

    check('/health ok', True)
    code, body = get(f'{base}/now')
    st = json.loads(body)
    check('/now 스키마', {'seq', 'status', 'window_sec'} <= set(st), str(sorted(st)[:5]))

    req = urllib.request.Request(f'{base}/detect?window=3', data=wav_bytes(),
                                 headers={'Content-Type': 'audio/wav'}, method='POST')
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    check('/detect 응답', r.status == 200 and 'lang' in d and 'margin' in d,
          f"{d.get('label')} margin={d.get('margin')}")
    check('/detect 창 반영', d['audio_sec'] <= 3.01, str(d['audio_sec']))

    # /events — 1줄 이상 받으면 성공 (2초 대기)
    try:
        with urllib.request.urlopen(f'{base}/events', timeout=8) as r:
            line = r.readline()
        check('/events 스트림', line.startswith(b'{'), line[:60].decode('utf-8', 'ignore'))
    except Exception as e:
        check('/events 스트림', False, f'{type(e).__name__}')

    p.terminate()
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
