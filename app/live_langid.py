#!/usr/bin/env python
"""eg2-langid 라이브 창 — 지금 PC에서 나는 소리의 언어를 표시 (Tkinter, 항상 위).

실행: python app/live_langid.py
  · 판정은 eg2_langid.engine.LiveEngine 이 담당 (캡처·임베딩·언어/음악 판정)
  · 이 파일은 표시만 담당 — 우클릭: 종료 · 일시정지, 드래그: 창 이동
  · 같은 엔진을 API/CLI가 그대로 쓴다 (eg2_langid.server / eg2_langid.analyze)

의존: numpy, pyaudiowpatch, tkinter, + 임베딩 모델(별도 배치)
"""
from __future__ import annotations

import os
import queue
import sys
import time
import tkinter as tk

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from eg2_langid.capture import LoopbackCapture, SR   # noqa: E402,F401  (테스트 호환 재노출)
from eg2_langid.engine import EngineConfig, LiveEngine  # noqa: E402

WINDOW_SEC = float(os.environ.get('EG2_WINDOW', '5.0'))
HOLD_SEC = float(os.environ.get('EG2_HOLD', '20'))   # 음악 구간에서 최근 언어를 유지하는 시간(초)
SHOW_LATENCY = os.environ.get('EG2_LATENCY', '1') != '0'

DISPLAY = {                      # 한국어 표기 (OS가 한국어일 때)
    'ko_kr': '한국어', 'en_us': '영어', 'ja_jp': '일본어', 'cmn_hans_cn': '중국어(간체)',
    'de_de': '독일어', 'es_419': '스페인어', 'fr_fr': '프랑스어', 'it_it': '이탈리아어',
    'pt_br': '포르투갈어', 'nl_nl': '네덜란드어', 'pl_pl': '폴란드어', 'ru_ru': '러시아어',
    'sv_se': '스웨덴어', 'uk_ua': '우크라이나어', 'hi_in': '힌디어', 'ar_eg': '아랍어',
    'th_th': '태국어', 'tr_tr': '튀르키예어', 'vi_vn': '베트남어', 'fil_ph': '필리핀어',
    'id_id': '인도네시아어', 'ms_my': '말레이어', 'ur_pk': '우르두어', 'da_dk': '덴마크어',
    'fa_ir': '페르시아어', 'cs_cz': '체코어', 'hr_hr': '크로아티아어', 'hy_am': '아르메니아어',
    'sw_ke': '스와힐리어',
}

DISPLAY_EN = {                   # English labels (default)
    'ko_kr': 'Korean', 'en_us': 'English', 'ja_jp': 'Japanese', 'cmn_hans_cn': 'Chinese',
    'de_de': 'German', 'es_419': 'Spanish', 'fr_fr': 'French', 'it_it': 'Italian',
    'pt_br': 'Portuguese', 'nl_nl': 'Dutch', 'pl_pl': 'Polish', 'ru_ru': 'Russian',
    'sv_se': 'Swedish', 'uk_ua': 'Ukrainian', 'hi_in': 'Hindi', 'ar_eg': 'Arabic',
    'th_th': 'Thai', 'tr_tr': 'Turkish', 'vi_vn': 'Vietnamese', 'fil_ph': 'Filipino',
    'id_id': 'Indonesian', 'ms_my': 'Malay', 'ur_pk': 'Urdu', 'da_dk': 'Danish',
    'fa_ir': 'Persian', 'cs_cz': 'Czech', 'hr_hr': 'Croatian', 'hy_am': 'Armenian',
    'sw_ke': 'Swahili',
}

STR = {
    'en': {
        'preparing': 'starting…', 'silent': 'silent', 'unknown': 'unknown',
        'error': 'error', 'pause': 'Pause / resume', 'quit': 'Quit',
        'wait_speech': 'waiting for speech · embedding skipped',
        'wait_input': 'no input — idle',
        'wait_sound': 'waiting for sound…',
        'model_load': 'model loaded in {t:.1f}s', 'capture_start': 'capture started',
        'capture_fail': 'capture failed: {e}', 'detect_fail': 'detect failed: {e}',
        'urdu_group': ' (incl. Urdu)', 'candidates': 'candidates: {s}',
        'music': 'music', 'sfx': 'sound effect',
        'hold': 'music section · holding {lang}',
        'non_speech_sub': 'no speech detected · {k} {p:.0f}%',
        'latency': 'delay {w:.0f}s window + {p}ms',
        'preview_sub': 'checking · {w:.0f}s window (confirms at 5s)',
    },
    'ko': {
        'preparing': '준비 중…', 'silent': '무음', 'unknown': '판정 불가',
        'error': '오류', 'pause': '일시정지 / 재개', 'quit': '종료',
        'wait_speech': '발화 대기 중 · 임베딩 생략',
        'wait_input': '입력 없음 — 대기',
        'wait_sound': '소리 기다리는 중…',
        'model_load': '모델 로드 {t:.1f}s', 'capture_start': '캡처 시작',
        'capture_fail': '캡처 실패: {e}', 'detect_fail': '판정 실패: {e}',
        'urdu_group': ' (우르두 포함)', 'candidates': '후보 {s}',
        'music': '음악', 'sfx': '효과음',
        'hold': '음악 구간 · 최근 언어 유지',
        'non_speech_sub': '발화 없음 · {k} {p:.0f}%',
        'latency': '지연 {w:.0f}초 창 + {p}ms',
        'preview_sub': '확인 중 · {w:.0f}초 창 (5초에 확정)',
    },
}


def detect_ui_language() -> str:
    """OS UI 언어가 한국어면 'ko', 아니면 'en'. EG2_LANG=ko|en 으로 강제 가능."""
    force = os.environ.get('EG2_LANG', '').strip().lower()
    if force.startswith('ko'):
        return 'ko'
    if force.startswith('en'):
        return 'en'
    if sys.platform == 'win32':
        try:
            import ctypes
            langid = ctypes.windll.kernel32.GetUserDefaultUILanguage()
            return 'ko' if (langid & 0x3FF) == 0x12 else 'en'   # 0x12 = Korean
        except Exception:
            pass
    import locale
    try:
        loc = locale.getlocale()[0] or ''
    except Exception:
        loc = ''
    loc = loc or os.environ.get('LC_ALL') or os.environ.get('LANG') or ''
    return 'ko' if loc.lower().startswith('ko') else 'en'


class App:
    BG = '#14161a'; FG = '#e8eaed'; DIM = '#8b93a1'
    OK = '#4ade80'; WARN = '#fbbf24'; NONE = '#6b7280'; NS_COLOR = '#a78bfa'

    def __init__(self):
        self.lang = detect_ui_language()
        self.T = STR[self.lang]
        self.names = DISPLAY if self.lang == 'ko' else DISPLAY_EN
        self.last_lang = None        # 최근 언어 판정(표시 유지용)
        self.last_lang_t = 0.0
        self.silent_run = 0

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
        self.name = tk.Label(top, text=self.T['preparing'], font=('Segoe UI', 15, 'bold'),
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
        self.menu.add_command(label=self.T['pause'], command=self._toggle_pause)
        self.menu.add_separator()
        self.menu.add_command(label=self.T['quit'], command=self.quit)
        self._dx = self._dy = 0

        # 판정 엔진 (캡처 + 임베딩 + 판정) — UI와 분리
        self.eng = LiveEngine(EngineConfig.from_env(window_sec=WINDOW_SEC))
        if '--json' in sys.argv:          # stdout NDJSON (다른 프로그램이 읽어갈 수 있게)
            import json as _json

            def _sink(kind, payload):
                if kind not in ('result', 'silent', 'error'):
                    return
                ev = dict(self.eng.latest(), event=kind)
                print(_json.dumps(ev, ensure_ascii=False), flush=True)
            self.eng.on_event = _sink
        self.eng.start()
        self.root.after(300, self._poll)

    # ── Tk 이벤트
    def _drag_start(self, e):
        self._dx, self._dy = e.x_root - self.root.winfo_x(), e.y_root - self.root.winfo_y()

    def _drag_move(self, e):
        self.root.geometry(f'+{e.x_root - self._dx}+{e.y_root - self._dy}')

    def _menu(self, e):
        self.menu.tk_popup(e.x_root, e.y_root)

    def _toggle_pause(self):
        self.eng.pause(not self.eng.paused)

    def quit(self):
        self.eng.stop()
        self.root.destroy()

    # ── 메인 루프
    def _poll(self):
        try:
            while True:
                kind, payload = self.eng.q.get_nowait()
                if kind == 'result':
                    self._render(payload)
                elif kind == 'error':
                    self._show_error(payload['message'])
                elif kind == 'silent':
                    self._render_silent(payload['run'])
                elif kind == 'status':
                    key = payload.get('key')
                    if key == 'model_load':
                        self.sub.config(text=self.T['model_load'].format(t=payload['t']))
                    elif key == 'capture_start':
                        self.sub.config(text=self.T['capture_start'])
        except queue.Empty:
            pass
        self.root.after(150, self._poll)

    def _render(self, payload):
        r = payload['result']
        sec = payload['audio_sec']
        proc_ms = payload.get('process_ms', 0)
        preview = bool(payload.get('preview'))
        self.silent_run = 0
        if getattr(r, 'kind', 'speech') != 'speech':
            # 음악/효과음 구간이어도 최근에 언어가 나왔다면 잠시 유지 (인터뷰·게임 방송 UX)
            age = time.time() - self.last_lang_t
            if HOLD_SEC > 0 and self.last_lang and age < HOLD_SEC:
                self.dot.itemconfig(1, fill=self.NONE)
                self.name.config(text=self.last_lang, fg=self.DIM)
                self.conf.config(text=f'-{age:.0f}s')
                self._bar(0, self.NONE)
                self.sub.config(text=self.T['hold'].format(lang=self.last_lang))
                return
            color = self.NS_COLOR
            self.dot.itemconfig(1, fill=color)
            name = self.T.get(r.kind, r.kind)
            self.name.config(text=name, fg=self.FG)
            p = r.p_nonspeech or 0.0
            self.conf.config(text=f'{p*100:.0f}%')
            self._bar(min(p, 0.99), color)
            self.sub.config(text=self.T['non_speech_sub'].format(k=name, p=p * 100))
            return
        if r.unknown:
            self.dot.itemconfig(1, fill=self.WARN)
            self.name.config(text=self.T['unknown'], fg=self.DIM)
            self.conf.config(text=f'{sec:.0f}s')
            self._bar(0, self.WARN)
            top = ' · '.join(list(r.scores)[:2])
            self.sub.config(text=self.T['candidates'].format(s=top)[:34])
            return
        label = self.names.get(r.label.split('/')[0], r.label)
        self.last_lang, self.last_lang_t = label, time.time()
        if '/' in r.label:
            label += self.T['urdu_group']
        # 표시값 = 정답 확률(p_correct). 없으면 게이트 여유(confidence)로 폴백
        conf = r.p_correct if r.p_correct is not None else r.confidence
        if preview:                       # 미리보기 — 곧 확정으로 교체된다(흐린 표시)
            self.dot.itemconfig(1, fill=self.NONE)
            self.name.config(text=label, fg=self.DIM)
            self.conf.config(text=f'{min(conf, 0.99)*100:.0f}%')
            self._bar(min(conf, 0.99), self.NONE)
            self.sub.config(text=self.T['preview_sub'].format(w=sec))
            return
        color = self.OK if conf >= 0.5 else self.WARN
        self.dot.itemconfig(1, fill=color)
        self.name.config(text=label, fg=self.FG)
        self.conf.config(text=f'{min(conf, 0.99)*100:.0f}%')
        self._bar(min(conf, 0.99), color)
        if SHOW_LATENCY:
            self.sub.config(text=self.T['latency'].format(w=WINDOW_SEC, p=proc_ms))
        else:
            self.sub.config(text=f'margin {r.margin:.2f} · {sec:.0f}s window')

    def _render_silent(self, run_count):
        self.silent_run = run_count
        self.dot.itemconfig(1, fill=self.NONE)
        self.name.config(text=self.T['silent'], fg=self.DIM)
        self.conf.config(text='')
        self._bar(0, self.NONE)
        self.sub.config(text=self.T['wait_speech'] if run_count < 3 else self.T['wait_input'])

    def _bar(self, frac, color):
        w = self.bar.winfo_width() or 248
        self.bar.coords(self.bar_fill, 0, 0, max(0, int(w * frac)), 6)
        self.bar.itemconfig(self.bar_fill, fill=color)

    def _show_error(self, msg):
        self.dot.itemconfig(1, fill='#ef4444')
        self.name.config(text=self.T['error'], fg='#ef4444')
        self.sub.config(text=msg[:80])

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    App().run()
