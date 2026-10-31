#!/usr/bin/env python
"""eg2-langid 라이브 창 — 지금 PC에서 나는 소리의 언어를 표시 (Tkinter, 항상 위).

실행: python app/live_langid.py
  · WASAPI 루프백(스피커 출력)을 5초 링버퍼로 캡처
  · 1초마다 임베딩 → 언어/음악/불확실 표시
  · 우클릭: 종료 · 일시정지

의존: numpy, sounddevice(불필요), pyaudiowpatch, tkinter, + 임베딩 모델(별도 배치)
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
import tkinter as tk

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from eg2_langid import LanguageIdentifier          # noqa: E402
from eg2_langid.adapters.gemma2 import make_embedder  # noqa: E402

SR = 16000
WINDOW_SEC = 5.0
HOP_SEC = 1.0

DISPLAY = {
    'ko_kr': '한국어', 'en_us': '영어', 'ja_jp': '일본어', 'cmn_hans_cn': '중국어(간체)',
    'de_de': '독일어', 'es_419': '스페인어', 'fr_fr': '프랑스어', 'it_it': '이탈리아어',
    'pt_br': '포르투갈어', 'nl_nl': '네덜란드어', 'pl_pl': '폴란드어', 'ru_ru': '러시아어',
    'sv_se': '스웨덴어', 'uk_ua': '우크라이나어', 'hi_in': '힌디어', 'ar_eg': '아랍어',
    'th_th': '태국어', 'tr_tr': '튀르키예어', 'vi_vn': '베트남어', 'fil_ph': '필리핀어',
    'id_id': '인도네시아어', 'ms_my': '말레이어', 'ur_pk': '우르두어', 'da_dk': '덴마크어',
    'fa_ir': '페르시아어', 'cs_cz': '체코어', 'hr_hr': '크로아티아어', 'hy_am': '아르메니아어',
    'sw_ke': '스와힐리어',
}


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

    def run(self):
        pa = self.pa_mod.PyAudio()
        try:
            dev = pa.get_default_wasapi_loopback()
            self.device_name = dev['name']
            ch = max(1, min(2, dev['maxInputChannels']))
            rate = int(dev['defaultSampleRate'])
            stream = pa.open(format=self.pa_mod.paFloat32, channels=ch, rate=rate,
                             input=True, input_device_index=dev['index'],
                             frames_per_buffer=int(rate * 0.2))
            self.sr = rate
            with self.lock:                                     # 장치 레이트에 맞춰 버퍼 재할당
                self.buf = np.zeros(int(self.window_sec * rate), np.float32)
                self.new_samples = 0
            while self.running:
                raw = stream.read(int(rate * 0.2), exception_on_overflow=False)
                a = np.frombuffer(raw, dtype=np.float32)
                if ch > 1:
                    a = a.reshape(-1, ch).mean(axis=1)
                with self.lock:
                    n = len(a)
                    if n >= len(self.buf):
                        self.buf[:] = a[-len(self.buf):]
                    else:
                        self.buf = np.concatenate([self.buf[n:], a])
                    self.new_samples += n
            stream.stop_stream(); stream.close()
        except Exception as e:                      # 장치 없음/권한 등
            self.error = f'{type(e).__name__}: {e}'
        finally:
            pa.terminate()

    def snapshot(self, sec: float) -> np.ndarray:
        with self.lock:
            n = int(sec * self.sr)
            return self.buf[-n:].copy()

    def seconds_buffered(self) -> float:
        with self.lock:
            return min(len(self.buf), self.new_samples) / self.sr

    def resample_to(self, a: np.ndarray, target: int = SR) -> np.ndarray:
        if self.sr == target or len(a) == 0:
            return a.astype(np.float32)
        n = int(len(a) * target / self.sr)
        return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)


class App:
    BG = '#14161a'; FG = '#e8eaed'; DIM = '#8b93a1'
    OK = '#4ade80'; WARN = '#fbbf24'; NONE = '#6b7280'

    def __init__(self):
        self.li = LanguageIdentifier()
        self.embed = None
        self.embed_err = None
        self.cap = None
        self.paused = False
        self.q: queue.Queue = queue.Queue()

        self.root = tk.Tk()
        self.root.title('eg2-langid')
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)
        self.root.geometry('+40+40')
        self.root.configure(bg=self.BG)

        w = tk.Frame(self.root, bg=self.BG, padx=14, pady=10)
        w.pack()
        top = tk.Frame(w, bg=self.BG); top.pack(fill='x')
        self.dot = tk.Canvas(top, width=10, height=10, bg=self.BG, highlightthickness=0)
        self.dot.create_oval(1, 1, 9, 9, fill=self.NONE, outline='')
        self.dot.pack(side='left', pady=(4, 0))
        self.name = tk.Label(top, text='준비 중…', font=('Segoe UI', 15, 'bold'),
                             fg=self.FG, bg=self.BG)
        self.name.pack(side='left', padx=8)
        self.conf = tk.Label(top, text='', font=('Consolas', 10), fg=self.DIM, bg=self.BG)
        self.conf.pack(side='right')
        self.close_btn = tk.Label(top, text='\u2715', font=('Segoe UI', 10), fg='#5b6270',
                                  bg=self.BG, cursor='hand2', padx=4)
        self.close_btn.pack(side='right', padx=(6, 0), before=self.conf)   # 최우측에 배치
        self.close_btn.bind('<Button-1>', lambda e: (self.quit(), 'break')[1])
        self.close_btn.bind('<Enter>', lambda e: self.close_btn.config(fg='#ef4444'))
        self.close_btn.bind('<Leave>', lambda e: self.close_btn.config(fg='#5b6270'))

        self.bar = tk.Canvas(w, height=6, bg='#22262d', highlightthickness=0, width=248)
        self.bar.pack(fill='x', pady=(8, 4))
        self.bar_fill = self.bar.create_rectangle(0, 0, 0, 6, fill=self.OK, outline='')

        self.sub = tk.Label(w, text='', font=('Consolas', 9), fg=self.DIM, bg=self.BG,
                            anchor='w', justify='left')
        self.sub.pack(fill='x')

        for widget in (self.root, w, top, self.name, self.sub, self.bar):
            widget.bind('<Button-1>', self._drag_start)
            widget.bind('<B1-Motion>', self._drag_move)
            widget.bind('<Button-3>', self._menu)
        self.menu = tk.Menu(self.root, tearoff=0)
        self.menu.add_command(label='일시정지 / 재개', command=self._toggle_pause)
        self.menu.add_separator()
        self.menu.add_command(label='종료', command=self.quit)
        self._dx = self._dy = 0

        threading.Thread(target=self._load_model, daemon=True).start()
        self.root.after(300, self._poll)
        self.root.after(200, self._start_capture)

    # ── Tk 이벤트
    def _drag_start(self, e):
        self._dx, self._dy = e.x_root - self.root.winfo_x(), e.y_root - self.root.winfo_y()

    def _drag_move(self, e):
        self.root.geometry(f'+{e.x_root - self._dx}+{e.y_root - self._dy}')

    def _menu(self, e):
        self.menu.tk_popup(e.x_root, e.y_root)

    def _toggle_pause(self):
        self.paused = not self.paused

    def quit(self):
        if self.cap:
            self.cap.running = False
        self.root.destroy()

    # ── 백그라운드
    def _load_model(self):
        t0 = time.time()
        try:
            self.embed = make_embedder(threads=4)
            self.q.put(('status', f'모델 로드 {time.time()-t0:.1f}s'))
        except Exception as e:
            self.embed_err = f'{type(e).__name__}: {e}'
            self.q.put(('error', self.embed_err))

    def _start_capture(self):
        try:
            self.cap = LoopbackCapture()
            self.cap.start()
            self.q.put(('status', '캡처 시작'))
        except Exception as e:
            self.q.put(('error', f'캡처 실패: {e}'))

    # ── 메인 루프
    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == 'result':
                    self._render(payload)
                elif kind == 'error':
                    self._show_error(payload)
                elif kind == 'status':
                    self.sub.config(text=payload)
        except queue.Empty:
            pass
        self.root.after(150, self._poll)

    def _worker(self):
        """1초마다 창 하나 판정 (별도 스레드에서 호출)."""
        if self.embed is None or self.cap is None or self.paused:
            return
        if self.cap.seconds_buffered() < 1.0:
            self.q.put(('status', '소리 기다리는 중…'))
            return
        a = self.cap.snapshot(WINDOW_SEC)
        a = self.cap.resample_to(a, SR)
        try:
            r = self.li.detect_audio(a, embed_fn=self.embed, window_sec=WINDOW_SEC)
        except Exception as e:
            self.q.put(('error', f'판정 실패: {e}'))
            return
        self.q.put(('result', (r, len(a) / SR)))

    def _render(self, payload):
        r, sec = payload
        if r.unknown:
            self.dot.itemconfig(1, fill=self.WARN)
            self.name.config(text='판정 불가', fg=self.DIM)
            self.conf.config(text=f'{sec:.0f}초')
            self._bar(0, self.WARN)
            top = ' · '.join(list(r.scores)[:2])
            self.sub.config(text=f'후보 {top}'[:34])
            return
        label = DISPLAY.get(r.label.split('/')[0], r.label)
        if '/' in r.label:
            label += ' (우르두 포함)'
        conf = r.confidence
        color = self.OK if conf >= 0.5 else self.WARN
        self.dot.itemconfig(1, fill=color)
        self.name.config(text=label, fg=self.FG)
        self.conf.config(text=f'{min(conf, 0.99)*100:.0f}%')
        self._bar(min(conf, 0.99), color)
        self.sub.config(text=f'margin {r.margin:.2f} · {sec:.0f}s window')

    def _bar(self, frac, color):
        w = self.bar.winfo_width() or 248
        self.bar.coords(self.bar_fill, 0, 0, max(0, int(w * frac)), 6)
        self.bar.itemconfig(self.bar_fill, fill=color)

    def _show_error(self, msg):
        self.dot.itemconfig(1, fill='#ef4444')
        self.name.config(text='오류', fg='#ef4444')
        self.sub.config(text=msg[:80])

    def run(self):
        def tick():
            threading.Thread(target=self._worker, daemon=True).start()
            self.root.after(int(HOP_SEC * 1000), tick)
        self.root.after(1500, tick)
        self.root.mainloop()


if __name__ == '__main__':
    App().run()
