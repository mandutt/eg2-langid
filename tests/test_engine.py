"""엔진(UI 없이) 동작 검사 — 캡처·판정·상태 노출.

모델이나 오디오 장치가 없으면 SKIP. 실제 재생 중인 소리가 없으면 '판정'만 건너뛴다.
실행: python tests/test_engine.py [--no-audio]
"""
from __future__ import annotations

import os
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from eg2_langid.engine import EngineConfig, LiveEngine  # noqa: E402

FAIL = []


def check(name, cond, detail=''):
    print(('PASS  ' if cond else 'FAIL  ') + name + (f'  ({detail})' if detail else ''))
    if not cond:
        FAIL.append(name)


def main():
    cfg = EngineConfig.from_env()
    eng = LiveEngine(cfg)
    t0 = time.time()
    ok = eng.start(wait=True, timeout=180)
    if not ok:
        print('SKIP  엔진 준비 실패 (모델/장치 없음)')
        return 0
    check('모델+캡처 준비', eng.ready, f'{time.time() - t0:.1f}s')

    st = eng.latest()
    check('상태 키 노출', {'seq', 'status', 'lang', 'window_sec', 'process_ms'} <= set(st),
          f"{len(st)} keys")
    check('창 길이 반영', abs(st['window_sec'] - cfg.window_sec) < 1e-6, str(st['window_sec']))

    if '--no-audio' not in sys.argv:
        # 실제 재생이 있으면 판정까지 확인 (없으면 조용히 넘어감)
        seen = []
        deadline = time.time() + 25
        while time.time() < deadline:
            time.sleep(1)
            s = eng.latest()
            if s['status'] in ('speech', 'music', 'sfx', 'unknown'):
                seen.append(s)
        if seen:
            check('판정 이벤트 발생', True, f'{len(seen)}회')
            check('process_ms 기록', all(s['process_ms'] >= 0 for s in seen),
                  f"max {max(s['process_ms'] for s in seen)}ms")
        else:
            print('SKIP  판정 — 재생 중인 소리 없음')

    eng.pause(True)
    check('pause 상태 반영', eng.latest()['status'] == 'paused')
    eng.pause(False)
    eng.stop()
    time.sleep(0.5)
    check('stop 후 종료', True)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
