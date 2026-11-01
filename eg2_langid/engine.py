"""실시간 판정 엔진 — 캡처·판정·상태를 담당하고 UI/API는 구독만 한다."""
from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import LanguageIdentifier
from .capture import LoopbackCapture

SR = 16000


@dataclass
class EngineConfig:
    """위젯/API가 공유하는 실행 설정. 기본값은 환경변수에서 온다."""
    window_sec: float = 5.0          # 판정 창 길이 — 지연의 하한
    hop_sec: float = 2.0             # 판정 주기(발화 중)
    idle_hop_sec: float = 4.0        # 판정 주기(무음)
    silence_rms: float = 5e-4        # 이보다 조용하면 임베딩 생략
    threads: int = 2                 # ONNX intra-op 스레드
    hybrid: bool = False             # 17특징(느림)
    verify_sec: float = 6.0          # 음악 판정 시 재확인 시작 간격
    verify_max: float = 20.0         # 백오프 상한
    verify_margin: float = 0.4       # 재확인에서 언어를 채택할 margin
    nomusic: bool = False            # 음악/효과음 판정 끔
    gap_sec: float = 0.4             # 이만큼 새 오디오가 없으면 무음으로 간주

    @classmethod
    def from_env(cls, **overrides) -> 'EngineConfig':
        env = os.environ
        cfg = cls(
            window_sec=float(env.get('EG2_WINDOW', 5.0)),
            hop_sec=float(env.get('EG2_HOP', 2.0)),
            idle_hop_sec=float(env.get('EG2_IDLE_HOP', 4.0)),
            silence_rms=float(env.get('EG2_SILENCE_RMS', 5e-4)),
            threads=int(env.get('EG2_THREADS', 2)),
            hybrid=env.get('EG2_HYBRID', '0') == '1',
            verify_sec=float(env.get('EG2_VERIFY', 6.0)),
            verify_max=float(env.get('EG2_VERIFY_MAX', 20.0)),
            verify_margin=float(env.get('EG2_VERIFY_MARGIN', 0.4)),
            nomusic=env.get('EG2_NOMUSIC', '0') == '1',
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg


@dataclass
class State:
    """API가 돌려주는 최신 상태."""
    seq: int = 0
    ts: float = 0.0                  # 판정 시각(unix)
    status: str = 'starting'         # starting | silent | speech | music | sfx | error | paused
    lang: str | None = None          # 'ko_kr' 등
    label: str | None = None         # 표시용
    p_correct: float | None = None
    confidence: float = 0.0
    margin: float = 0.0
    kind: str = 'speech'
    window_sec: float = 5.0
    process_ms: int = 0              # 이번 판정에 걸린 계산 시간
    silent_run: int = 0
    error: str | None = None
    device: str = ''

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d['ts'] = round(d['ts'], 3)
        return d


class LiveEngine:
    """루프백 캡처 → 주기 판정 → 큐/콜백으로 결과 발행.

    사용:
        eng = LiveEngine()
        eng.start()               # 모델 로드 + 캡처 + 판정 루프
        while True:
            kind, payload = eng.q.get()
        eng.stop()
    """

    def __init__(self, cfg: EngineConfig | None = None,
                 on_event: Callable[[str, object], None] | None = None):
        self.cfg = cfg or EngineConfig.from_env()
        self.li = LanguageIdentifier()
        if self.cfg.nomusic:
            self.li.ns.ok = False
        self.embed = None
        self.vad = None
        self.cap: LoopbackCapture | None = None
        self.q: queue.Queue = queue.Queue()
        self.on_event = on_event
        self.paused = False
        self.silent_run = 0
        self.state = State(window_sec=self.cfg.window_sec)
        self._verify_next = 0.0
        self._verify_delay = self.cfg.verify_sec
        self._last_kind = 'speech'
        self._last_samples = 0
        self._cpu_prev: float | None = None
        self._stop = False
        self._ready = threading.Event()

    # ── 생명주기
    def start(self, wait: bool = False, timeout: float = 120.0) -> bool:
        """모델 로드·캡처 시작. wait=True면 준비될 때까지 블록."""
        threading.Thread(target=self._boot, daemon=True).start()
        if wait:
            return self._ready.wait(timeout)
        return True

    def stop(self):
        self._stop = True
        if self.cap is not None:
            self.cap.running = False

    @property
    def ready(self) -> bool:
        return self.embed is not None and self.cap is not None

    def _boot(self):
        t0 = time.time()
        try:
            from .adapters.gemma2 import make_embedder
            self.embed = make_embedder(threads=self.cfg.threads)
        except Exception as e:                       # 모델 없음 → 오류 상태로 대기
            self._emit('error', {'message': f'model: {type(e).__name__}: {e}',
                                 'fatal': True, 'state': 'error'})
            self._ready.set()
            return
        try:
            from .adapters.silero_vad import make_vad
            self.vad = make_vad()
        except Exception:
            self.vad = None
        self._emit('status', {'key': 'model_load', 't': time.time() - t0})
        try:
            self.cap = LoopbackCapture(window_sec=self.cfg.window_sec)
            self.cap.start()
            self.state.device = self.cap.device_name
            self._emit('status', {'key': 'capture_start'})
        except Exception as e:
            self._emit('error', {'message': f'capture: {type(e).__name__}: {e}',
                                 'fatal': True, 'state': 'error'})
            self._ready.set()
            return
        self._ready.set()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop:
            try:
                self._worker()
            except Exception as e:
                self._emit('error', {'message': f'{type(e).__name__}: {e}',
                                     'fatal': False, 'state': 'error'})
            time.sleep(self.hop)

    @property
    def hop(self) -> float:
        return self.cfg.idle_hop_sec if self.silent_run >= 2 else self.cfg.hop_sec

    # ── 판정 1회
    def _worker(self):
        if self.embed is None or self.cap is None or self.paused:
            return
        self._watchdog()
        if self.cap.seconds_buffered() < 1.0:        # 아직 오디오가 흐르지 않음
            self._silent('wait_input')
            return
        a = self.cap.snapshot(self.cfg.window_sec)
        a = self.cap.resample_to(a, SR)
        if len(a) < SR // 2:
            return
        new = self.cap.new_samples
        if new - self._last_samples < int(self.cap.sr * self.cfg.gap_sec):
            self._silent('wait_speech')              # 재생 중단·정지 → 임베딩 생략
            return
        self._last_samples = new
        rms = float(np.sqrt(np.mean(a * a)))
        if os.environ.get('EG2_DEBUG'):
            print(f'[tick] rms={rms:.6f} silent={"Y" if rms < self.cfg.silence_rms else "N"} '
                  f'buf={len(a)/SR:.1f}s thr={self.cfg.silence_rms}', flush=True)
        if rms < self.cfg.silence_rms:               # 무음 → 임베딩 생략 (CPU 0)
            self._silent('wait_speech')
            return
        self.silent_run = 0
        t0 = time.perf_counter()
        now = time.time()
        was_music = self._last_kind != 'speech'
        verify = self.cfg.verify_sec > 0 and now >= self._verify_next
        if verify:
            self._verify_next = now + max(1.0, self._verify_delay)
        r = self.li.detect_audio(a, embed_fn=self.embed, window_sec=self.cfg.window_sec,
                                 vad_fn=self.vad, use_hybrid=self.cfg.hybrid,
                                 verify_music=verify, verify_margin=self.cfg.verify_margin)
        if os.environ.get('EG2_DEBUG') and verify:
            print(f'[verify] t={now:.0f}s delay={self._verify_delay:.0f}s '
                  f'was_music={was_music} → {r.kind}', flush=True)
        if verify and was_music:
            if r.kind == 'speech':
                self._verify_delay = self.cfg.verify_sec      # 발화를 건졌다 → 다시 촘촘히
                self._verify_next = now                        # 다음 창에서도 즉시 확인
            else:
                self._verify_delay = min(self.cfg.verify_max,
                                         max(self.cfg.verify_sec, self._verify_delay * 2))
        self._last_kind = r.kind
        self._emit('result', {'result': r, 'audio_sec': len(a) / SR,
                              'process_ms': int((time.perf_counter() - t0) * 1000),
                              'status': self._status_of(r)})

    def _silent(self, status: str):
        self.silent_run += 1
        self._emit('silent', {'run': self.silent_run, 'status': status})

    def _watchdog(self):
        """무음인데 프로세스가 CPU를 태우면(=루프백 스핀 등) 스트림을 재개방한다."""
        cpu_now = time.process_time()
        if self._cpu_prev is not None and self.silent_run >= 2:
            if cpu_now - self._cpu_prev > 2.0:
                self.cap.request_reopen()
                print(f'[cap] watchdog: silent but {cpu_now - self._cpu_prev:.1f}s CPU → reopen',
                      flush=True)
        self._cpu_prev = cpu_now

    @staticmethod
    def _status_of(r) -> str:
        if getattr(r, 'kind', 'speech') != 'speech':
            return r.kind
        if r.unknown:
            return 'unknown'
        return 'speech'

    # ── 발행
    def _emit(self, kind: str, payload):
        st = self.state
        st.seq += 1
        st.ts = time.time()
        st.window_sec = self.cfg.window_sec
        st.error = None
        if kind == 'result':
            r = payload['result']
            st.kind = getattr(r, 'kind', 'speech')
            st.status = payload['status']
            st.lang = r.lang
            st.label = r.label
            st.p_correct = r.p_correct
            st.confidence = r.confidence
            st.margin = r.margin
            st.process_ms = payload['process_ms']
            st.silent_run = 0
        elif kind == 'silent':
            st.status = 'silent'
            st.silent_run = payload['run']
            st.kind = 'speech'
            st.lang = None
            st.label = None
            st.p_correct = None
            st.confidence = 0.0
        elif kind == 'status':
            st.device = self.cap.device_name if self.cap is not None else ''
        elif kind == 'error':
            st.status = 'error'
            st.error = payload['message']
        self.q.put((kind, payload))
        if self.on_event is not None:
            try:
                self.on_event(kind, payload)
            except Exception:
                pass

    # ── 조회/제어
    def latest(self) -> dict:
        if self.cap is not None and self.cap.device_name:
            self.state.device = self.cap.device_name
        return self.state.to_dict()

    def pause(self, on: bool = True):
        self.paused = on
        self.state.status = 'paused' if on else 'silent'
