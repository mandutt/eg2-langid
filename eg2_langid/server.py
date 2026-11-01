"""실시간 판정 API — HTTP(JSON) + stdout(NDJSON).

    python -m eg2_langid.server                       # http://127.0.0.1:8730
    python -m eg2_langid.server --json                # NDJSON을 stdout으로도
    python -m eg2_langid.server --host 100.x.y.z      # Tailscale 등 외부 바인드
    python -m eg2_langid.server --no-http --json      # 순수 파이프 모드

엔드포인트:
    GET  /health   {ok, ready, device, window_sec}
    GET  /now      최신 판정 1건
    GET  /events   NDJSON 스트림 (판정이 바뀔 때마다 한 줄, keep-alive)
    POST /detect   요청 본문 = 오디오 파일 바이트(wav/mp3/m4a…) → 한 번 판정
                   ?window=5  (선택) 판정 창 길이
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from .engine import EngineConfig, LiveEngine

DEFAULT_PORT = 8730


class Bus:
    """엔진 이벤트를 여러 클라이언트가 각자 위치에서 읽도록."""

    def __init__(self, maxlen: int = 500):
        self.items: list[dict] = []
        self.cv = threading.Condition()
        self.maxlen = maxlen
        self.closed = False

    def push(self, ev: dict):
        with self.cv:
            self.items.append(ev)
            if len(self.items) > self.maxlen:
                del self.items[:len(self.items) - self.maxlen]
            self.cv.notify_all()

    def read_from(self, idx: int, timeout: float = 20.0) -> tuple[list[dict], int]:
        with self.cv:
            if idx >= len(self.items) and not self.closed:
                self.cv.wait(timeout)
            out = self.items[idx:]
            return out, idx + len(out)


def _json_line(obj: dict) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + '\n').encode('utf-8')


def make_handler(engine: LiveEngine, bus: Bus):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, fmt, *args):      # 조용히
            if os.environ.get('EG2_DEBUG'):
                sys.stderr.write('[http] ' + (fmt % args) + '\n')

        def _send(self, code: int, body: bytes, ctype: str = 'application/json'):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == '/health':
                st = engine.latest()
                self._send(200, json.dumps({
                    'ok': True, 'ready': engine.ready, 'device': st.get('device', ''),
                    'window_sec': st.get('window_sec'), 'version': 1,
                }, ensure_ascii=False).encode('utf-8'))
            elif u.path == '/now':
                self._send(200, json.dumps(engine.latest(), ensure_ascii=False).encode('utf-8'))
            elif u.path == '/events':
                self.send_response(200)
                self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Transfer-Encoding', 'chunked')
                self.end_headers()
                idx = 0
                try:
                    # 접속 즉시 현재 상태 1줄
                    snap = dict(engine.latest(), event='snapshot')
                    self._chunk(_json_line(snap))
                    idx = len(bus.items)
                    while True:
                        items, idx = bus.read_from(idx)
                        for ev in items:
                            self._chunk(_json_line(ev))
                        if not items:
                            self._chunk(b'')          # keep-alive
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return
            else:
                self._send(404, b'{"error":"not found"}')

        def _chunk(self, data: bytes):
            if not data:
                self.wfile.write(b'0\r\n\r\n' if False else b'')   # noop keep-alive
                self.wfile.flush()
                return
            self.wfile.write(f'{len(data):X}\r\n'.encode() + data + b'\r\n')
            self.wfile.flush()

        def do_POST(self):
            u = urlparse(self.path)
            if u.path != '/detect':
                self._send(404, b'{"error":"not found"}')
                return
            n = int(self.headers.get('Content-Length') or 0)
            if n <= 0:
                self._send(400, b'{"error":"empty body"}')
                return
            data = self.rfile.read(n)
            q = parse_qs(u.query)
            win = float(q.get('window', ['5'])[0])
            if engine.embed is None:
                self._send(503, b'{"error":"model not ready"}')
                return
            suffix = '.bin'
            for ext in ('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.opus', '.webm', '.mp4'):
                if data[:4] == b'RIFF' and ext == '.wav':
                    suffix = ext
                    break
            path = None
            try:
                fd, path = tempfile.mkstemp(suffix=suffix)
                with os.fdopen(fd, 'wb') as f:
                    f.write(data)
                audio = _decode(path)
                if len(audio) < 16000:
                    self._send(400, b'{"error":"audio too short"}')
                    return
                seg = audio[-int(win * 16000):]
                r = engine.li.detect(engine.embed(np.ascontiguousarray(seg)),
                                     audio_sec=len(seg) / 16000)
                out = r.to_dict()
                out['request'] = {'window_sec': win, 'bytes': n}
                self._send(200, json.dumps(out, ensure_ascii=False).encode('utf-8'))
            except Exception as e:
                self._send(500, json.dumps({'error': f'{type(e).__name__}: {e}'},
                                           ensure_ascii=False).encode('utf-8'))
            finally:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

    return Handler


def _decode(path: str) -> np.ndarray:
    from .analyze import decode
    return decode(path)


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog='eg2_langid.server')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=DEFAULT_PORT)
    ap.add_argument('--json', action='store_true', help='NDJSON을 stdout으로도 출력')
    ap.add_argument('--no-http', action='store_true', help='HTTP 없이 stdout만')
    ap.add_argument('--window', type=float, default=None)
    args = ap.parse_args(argv)

    over = {'window_sec': args.window} if args.window else {}
    engine = LiveEngine(EngineConfig.from_env(**over))
    bus = Bus()
    out_lock = threading.Lock()

    def on_event(kind, payload):
        if kind not in ('result', 'silent', 'error'):
            return
        ev = dict(engine.latest(), event=kind, ts=time.time())
        bus.push(ev)
        if args.json:
            with out_lock:
                sys.stdout.write(_json_line(ev).decode('utf-8'))
                sys.stdout.flush()

    engine.on_event = on_event
    engine.start()

    if not args.no_http:
        srv = ThreadingHTTPServer((args.host, args.port), make_handler(engine, bus))
        srv.daemon_threads = True
        print(f'eg2-langid API  http://{args.host}:{args.port}  '
              f'(/now /events /detect /health)', file=sys.stderr, flush=True)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            engine.stop()
            srv.server_close()
    else:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            engine.stop()
    return 0


if __name__ == '__main__':
    sys.exit(main())
