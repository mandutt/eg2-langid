"""파일 분석 — 구간 병합/표시 로직 검사.

실제 오디오 통합 검사는 EG2_TEST_AUDIO=<wav경로> 를 주면 수행한다.
실행: python tests/test_analyze.py
"""
from __future__ import annotations

import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from eg2_langid.analyze import (Analysis, Segment, _smooth, _ts,  # noqa: E402
                                format_text)

FAIL = []


def check(name, cond, detail=''):
    print(('PASS  ' if cond else 'FAIL  ') + name + (f'  ({detail})' if detail else ''))
    if not cond:
        FAIL.append(name)


def main():
    check('_ts 60초', _ts(61) == '1:01', _ts(61))
    check('_ts 1시간', _ts(3725) == '1:02:05', _ts(3725))

    # 짧은 뒤집기 제거: ko ko en ko ko → 가운데 en 제거
    s = _smooth(['ko_kr', 'ko_kr', 'en_us', 'ko_kr', 'ko_kr'], 3)
    check('스무딩 — 1창 뒤집기 제거', s[2] == 'ko_kr', str(s))
    s2 = _smooth(['ko_kr', 'en_us', 'en_us'], 3)
    check('스무딩 — 실제 전환 유지', s2[1] == 'en_us', str(s2))

    a = Analysis(path='x.wav', duration=40.0, window_sec=5.0, hop_sec=1.0, lang='ko',
                 segments=[Segment(0, 12, 'speech', 'ko_kr', 'ko_kr', 0.97, 12),
                           Segment(12, 22, 'music'),
                           Segment(22, 34, 'speech', 'en_us', 'en_us', 0.95, 12),
                           Segment(34, 37, 'silence'),
                           Segment(37, 40, 'unknown')],
                 proportions={'ko_kr': 0.5, 'en_us': 0.5}, speech_sec=24.0,
                 music_sec=10.0, silence_sec=3.0, unknown_sec=3.0)
    txt = format_text(a)
    check('표시 — 언어 구간', '한국어' in txt and '영어' in txt)
    check('표시 — 음악 구간', '[음악/효과음]' in txt)
    check('표시 — 무음 구간', '[무음]' in txt)
    check('표시 — 비중', '비중(발화 구간만)' in txt and '50.0%' in txt)

    d = a.to_dict()
    check('JSON — 구간 수', len(d['segments']) == 5, str(len(d['segments'])))
    check('JSON — 직렬화', json.dumps(d, ensure_ascii=False)[:1] == '{')

    audio = os.environ.get('EG2_TEST_AUDIO')
    if audio and os.path.exists(audio):
        from eg2_langid.analyze import analyze_file
        res = analyze_file(audio, hop=2.0, lang='ko')
        check('통합 — 구간 생성', len(res.segments) > 0, f'{len(res.segments)}구간')
        tot = res.speech_sec + res.unknown_sec + res.silence_sec + res.music_sec
        check('통합 — 시간 합계', abs(tot - res.duration) < res.hop_sec * 2, f'{tot:.1f}/{res.duration:.1f}')
        if res.proportions:
            check('통합 — 비중 합 1', abs(sum(res.proportions.values()) - 1) < 1e-3,
                  str(res.proportions))
    else:
        print('SKIP  통합 검사 — EG2_TEST_AUDIO 미지정')

    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
