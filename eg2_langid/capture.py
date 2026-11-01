"""WASAPI 루프백 캡처 — 최근 N초 링버퍼. UI/판정과 독립."""
from __future__ import annotations

import os
import threading
import time

import numpy as np

SR = 16000
WINDOW_SEC = 5.0


class LoopbackCapture(threading.Thread):
    """WASAPI 루프백 캡처 → 최근 window_sec만 유지하는 링버퍼."""

    def __init__(self, window_sec: float = WINDOW_SEC, sr: int = SR):
        super().__init__(daemon=True)
        import pyaudiowpatch as pyaudio
        self.pa_mod = pyaudio
        self.sr = sr
        self.window_sec = window_sec
        self.buf = np.zeros(int(window_sec * sr), np.float32)   # 실제 레이트 확정 후 재할당
        self.lock = threading.Lock()
        self.new_samples = 0
        self.running = True
        self.error: str | None = None
        self.device_name = ''
        self._reopen = False
        self._w = 0

    def run(self):
        pa = self.pa_mod.PyAudio()
        stream = None

        def _open():
            dev = pa.get_default_wasapi_loopback()
            self.device_name = dev['name']
            ch = max(1, min(2, dev['maxInputChannels']))
            rate = int(dev['defaultSampleRate'])
            s = pa.open(format=self.pa_mod.paFloat32, channels=ch, rate=rate,
                        input=True, input_device_index=dev['index'],
                        frames_per_buffer=int(rate * 0.2))
            self.sr = rate
            with self.lock:                             # 장치 레이트에 맞춰 버퍼 재할당
                self.buf = np.zeros(int(self.window_sec * rate), np.float32)
                self.new_samples = 0
                self._w = 0
            return s, ch, rate

        try:
            stream, ch, rate = _open()
            block = int(rate * 0.2)
            while self.running:
                if self._reopen:                        # 워치독이 요청 → 스트림 재개방
                    self._reopen = False
                    try:
                        stream.stop_stream(); stream.close()
                    except Exception:
                        pass
                    time.sleep(0.5)
                    stream, ch, rate = _open()
                    block = int(rate * 0.2)
                    print('[cap] stream reopened', flush=True)
                t0 = time.perf_counter()
                try:
                    raw = stream.read(block, exception_on_overflow=False)
                except Exception:
                    time.sleep(0.2)
                    continue
                a = np.frombuffer(raw, dtype=np.float32)
                if ch > 1:
                    a = a.reshape(-1, ch).mean(axis=1)
                with self.lock:                          # 인플레이스 링버퍼(재할당 없음)
                    n = len(a)
                    if n >= len(self.buf):
                        self.buf[:] = a[-len(self.buf):]
                        self._w = 0
                    else:
                        end = self._w + n
                        if end <= len(self.buf):
                            self.buf[self._w:end] = a
                        else:
                            k = len(self.buf) - self._w
                            self.buf[self._w:] = a[:k]
                            self.buf[:end - len(self.buf)] = a[k:]
                        self._w = end % len(self.buf)
                    self.new_samples += n
                # 무음일 때 WASAPI가 즉시 반환하면 이 루프가 CPU를 태운다 → 최소 주기 보장
                dt = time.perf_counter() - t0
                if dt < 0.2:
                    time.sleep(0.2 - dt)
            stream.stop_stream(); stream.close()
        except Exception as e:                      # 장치 없음/권한 등
            self.error = f'{type(e).__name__}: {e}'
        finally:
            pa.terminate()

    def request_reopen(self):
        """스핀/장치 먹통 의심 시 스트림 재개방 요청."""
        self._reopen = True

    def snapshot(self, sec: float) -> np.ndarray:
        with self.lock:
            n = min(int(sec * self.sr), len(self.buf))
            if self._w == 0:
                return self.buf[-n:].copy()
            if self._w >= n:
                return self.buf[self._w - n:self._w].copy()
            out = np.concatenate([self.buf[self._w:], self.buf[:self._w]])
            return out[-n:].copy()

    def seconds_buffered(self) -> float:
        with self.lock:
            return min(len(self.buf), self.new_samples) / self.sr

    def resample_to(self, a: np.ndarray, target: int = SR) -> np.ndarray:
        if self.sr == target or len(a) == 0:
            return a.astype(np.float32)
        n = int(len(a) * target / self.sr)
        return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)
