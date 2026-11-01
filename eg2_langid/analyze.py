"""파일 전체 언어 분석 — 구간별 언어 + 비중.

실시간과 달리 음악/효과음 분류를 하지 않는다(발화가 음악으로 오판되는 문제 회피).
대신 ① 디지털 무음은 건너뛰고 ② 확신이 낮은 창은 '판정 불가'로 남긴다.

사용:
    from eg2_langid.analyze import analyze_file, format_text
    a = analyze_file('interview.mp4')
    print(format_text(a))

CLI:
    python -m eg2_langid.analyze interview.mp4 --json out.json [--hop 1] [--window 5]
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

SR = 16000
SILENCE_RMS = 5e-4

LABELS_KO = {
    'ko_kr': '한국어', 'en_us': '영어', 'ja_jp': '일본어', 'cmn_hans_cn': '중국어',
    'de_de': '독일어', 'es_419': '스페인어', 'fr_fr': '프랑스어', 'it_it': '이탈리아어',
    'pt_br': '포르투갈어', 'nl_nl': '네덜란드어', 'pl_pl': '폴란드어', 'ru_ru': '러시아어',
    'sv_se': '스웨덴어', 'uk_ua': '우크라이나어', 'hi_in': '힌디어', 'ar_eg': '아랍어',
    'th_th': '태국어', 'tr_tr': '튀르키예어', 'vi_vn': '베트남어', 'fil_ph': '필리핀어',
    'id_id': '인도네시아어', 'ms_my': '말레이어', 'ur_pk': '우르두어', 'da_dk': '덴마크어',
    'fa_ir': '페르시아어', 'cs_cz': '체코어', 'hr_hr': '크로아티아어', 'hy_am': '아르메니아어',
    'sw_ke': '스와힐리어',
}
LABELS_EN = {
    'ko_kr': 'Korean', 'en_us': 'English', 'ja_jp': 'Japanese', 'cmn_hans_cn': 'Chinese',
    'de_de': 'German', 'es_419': 'Spanish', 'fr_fr': 'French', 'it_it': 'Italian',
    'pt_br': 'Portuguese', 'nl_nl': 'Dutch', 'pl_pl': 'Polish', 'ru_ru': 'Russian',
    'sv_se': 'Swedish', 'uk_ua': 'Ukrainian', 'hi_in': 'Hindi', 'ar_eg': 'Arabic',
    'th_th': 'Thai', 'tr_tr': 'Turkish', 'vi_vn': 'Vietnamese', 'fil_ph': 'Filipino',
    'id_id': 'Indonesian', 'ms_my': 'Malay', 'ur_pk': 'Urdu', 'da_dk': 'Danish',
    'fa_ir': 'Persian', 'cs_cz': 'Czech', 'hr_hr': 'Croatian', 'hy_am': 'Armenian',
    'sw_ke': 'Swahili',
}


@dataclass
class Segment:
    start: float
    end: float
    kind: str                 # speech | unknown | silence
    lang: str | None = None
    label: str | None = None
    mean_p: float | None = None     # 평균 p_correct
    windows: int = 0

    @property
    def dur(self) -> float:
        return round(self.end - self.start, 2)


@dataclass
class Analysis:
    path: str
    duration: float
    window_sec: float
    hop_sec: float
    segments: list[Segment] = field(default_factory=list)
    proportions: dict[str, float] = field(default_factory=dict)   # 발화 구간만, 코드→비율
    speech_sec: float = 0.0
    unknown_sec: float = 0.0
    silence_sec: float = 0.0
    music_sec: float = 0.0
    lang: str = 'ko'          # 출력 언어

    def to_dict(self) -> dict:
        d = asdict(self)
        d['duration'] = round(self.duration, 2)
        return d


def decode(path: str) -> np.ndarray:
    """ffmpeg로 16kHz mono float32 디코딩 (전체 메모리)."""
    exe = shutil.which('ffmpeg')
    if exe is None:
        cand = os.path.join(os.environ.get('LOCALAPPDATA', ''),
                            'hermes', 'tools', 'ffmpeg-9.0.1-win32-x64', 'bin', 'ffmpeg.exe')
        exe = cand if os.path.exists(cand) else 'ffmpeg'
    p = subprocess.run([exe, '-v', 'quiet', '-i', path, '-f', 'f32le', '-acodec', 'pcm_f32le',
                        '-ac', '1', '-ar', str(SR), '-'], capture_output=True)
    if p.returncode != 0 or not p.stdout:
        raise RuntimeError(f'디코딩 실패: {p.stderr.decode("utf-8", "ignore")[:200]}')
    return np.frombuffer(p.stdout, np.float32)


def _smooth(vals: list, k: int = 3) -> list:
    """짧은 뒤집기 제거 — 이웃 k개 다수결(가장 흔한 값)."""
    if k < 3 or len(vals) < 3:
        return vals
    out = list(vals)
    h = k // 2
    for i in range(h, len(vals) - h):
        win = [v for v in vals[i - h:i + h + 1] if v is not None]
        if win:
            out[i] = max(set(win), key=win.count)
    return out


def default_threads() -> int:
    """배치 기본 스레드 — 물리 코어 추정(logical//2), 2~6 범위.

    실측(12 logical = 6 physical): 1레인에서 threads 2→1.68, 4→3.14, 6→4.26, 8→4.37 창/s.
    6을 넘기면 이득이 멈추므로 상한을 6으로 둔다.
    """
    import os as _os
    logical = _os.cpu_count() or 4
    return max(2, min(6, logical // 2))


def analyze_file(path: str, window: float = 5.0, hop: float = 2.0,
                 min_seg: float = 2.0, threads: int | None = None, lang: str = 'ko',
                 music: str = 'mark', music_margin: float = 0.3,
                 progress: Callable[[float, float], None] | None = None,
                 embed: Callable | None = None, lanes: int | None = None) -> Analysis:
    """파일 전체를 창 단위로 판정해 구간·비중을 만든다.

    music='mark'(기본) — 음악/효과음 확률이 높아도 **언어 margin이 높으면 발화로 인정**한다.
        실측: 음악 구간 margin 0.02~0.11 vs 발화 0.62~0.85 → 배경음 깔린 인터뷰를 음악으로
        오판하던 문제를 피하면서 음악 구간은 [음악]으로 표시한다.
    music='off'  — 음악 판정을 아예 하지 않는다(음악 구간도 언어로 집계됨).
    music='gate' — 비발화 확률만 보고 음악으로 제외한다(실시간 기본과 같은 방식).
    """
    from . import LanguageIdentifier
    from .adapters.gemma2 import make_embedder

    if lanes is None:
        # 실측(threads 6): 1레인 4.09 · 2레인 5.59 · 3레인 6.02 · 4레인 6.44 · 5레인 6.09 ·
        # 6레인 6.91(정점) · 8레인 6.00 창/s, RSS +44→348MB. 물리 코어 수에서 꺾인다.
        lanes = max(1, min(6, int(os.environ.get('EG2_LANES', default_threads()))))
    audio = decode(path)
    dur = len(audio) / SR
    li = LanguageIdentifier()
    if embed is None:                       # 여러 파일을 처리할 때는 한 번만 만들어 넘긴다
        embed = make_embedder(threads=threads or default_threads())
    vad = None
    if music != 'off':
        try:
            from .adapters.silero_vad import make_vad
            vad = make_vad()
        except Exception:
            vad = None
    thr = (li._ns_thr['ns_vad_thr' if vad is not None else 'ns_novad_thr'])

    win = int(window * SR)
    step = int(hop * SR)
    rows = []          # (start, end, kind, lang, p)
    pos = 0
    last_end = 0.0     # 판정 구간은 겹치지 않게 이어 붙인다
    min_judge = int(min(window, 3.0) * SR)     # 창이 이보다 짧으면 판정 보류(파일 앞부분)

    # 1단계 — 판정 대상 창을 먼저 확정한다(무음 판정은 임베딩이 필요 없다).
    # 오디오는 이미 전부 메모리에 있으므로 조각은 (시작, 끝) 샘플 인덱스로만 들고 있는다.
    jobs = []          # (kind, start, end, (s0, s1) | None)
    while pos < len(audio):
        start = last_end
        end = min((pos + step) / SR, dur)
        block = audio[pos:pos + step]
        rms = float(np.sqrt(np.mean(block * block))) if len(block) else 0.0
        if rms < SILENCE_RMS or end - start < 0.2:
            if end > last_end:
                jobs.append(('silence', start, end, None))
                last_end = end
            pos += step
            continue
        s0, s1 = max(0, pos + step - win), pos + step
        if s1 - s0 < min_judge:                # 아직 비교할 오디오가 부족
            pos += step
            continue
        jobs.append(('judge', start, end, (s0, s1)))
        last_end = end
        pos += step

    # 2단계 — 임베딩만 병렬로(순서 보존). 스레드는 세션을 공유하므로 모델은 한 벌.
    todo = [j[3] for j in jobs if j[0] == 'judge']
    if todo and lanes and lanes > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(lanes, len(todo))) as ex:
            vectors = list(ex.map(lambda se: embed(np.ascontiguousarray(audio[se[0]:se[1]])), todo))
    else:
        vectors = [embed(np.ascontiguousarray(audio[s0:s1])) for s0, s1 in todo]
    vit = iter(vectors)

    # 3단계 — 판정은 순서대로(스무딩·구간 병합이 시간 순서를 요구한다).
    for kind, start, end, seg_idx in jobs:
        if kind == 'silence':
            rows.append((start, end, 'silence', None, None))
            if progress:
                progress(start, dur)
            continue
        vec = next(vit)
        r = li.detect(vec, audio_sec=(seg_idx[1] - seg_idx[0]) / SR)
        p_ns = None
        if music != 'off':
            p_ns, _, _ = li.ns.score(audio[seg_idx[0]:seg_idx[1]], vad, vec=None, hybrid=False)
        is_music = (music != 'off' and p_ns is not None and p_ns >= thr
                    and (music == 'gate' or r.margin < music_margin))
        if is_music:
            rows.append((start, end, 'music', None, None))
        elif r.unknown:
            rows.append((start, end, 'unknown', None, r.p_correct))
        else:
            rows.append((start, end, 'speech', r.label, r.p_correct))
        if progress:
            progress(start, dur)

    if not rows:
        return Analysis(path=path, duration=dur, window_sec=window, hop_sec=hop, lang=lang)

    # ── 스무딩: 이웃 3개 중앙값으로 짧은 뒤집기 제거
    keys = [(k if k != 'speech' else l) for _, _, k, l, _ in rows]
    keys = _smooth(keys, 3)
    rows = [(rows[i][0], rows[i][1],
             ('speech' if keys[i] not in ('unknown', 'silence', 'music') else keys[i]),
             (keys[i] if keys[i] not in ('unknown', 'silence', 'music') else None), rows[i][4])
            for i in range(len(rows))]

    # ── 구간 병합
    segs: list[Segment] = []
    for i, (start, end, kind, label, p) in enumerate(rows):
        if segs and segs[-1].kind == kind and segs[-1].lang == label:
            segs[-1].end = end
            segs[-1].windows += 1
            if p is not None:
                prev = segs[-1].mean_p or 0.0
                segs[-1].mean_p = round((prev * (segs[-1].windows - 1) + p) / segs[-1].windows, 3)
        else:
            segs.append(Segment(start=round(start, 2), end=round(end, 2), kind=kind,
                                lang=label, label=label, mean_p=p, windows=1))

    # ── 너무 짧은 구간은 이웃에 흡수
    merged: list[Segment] = []
    for s in segs:
        if merged and s.dur < min_seg:
            merged[-1].end = s.end
            merged[-1].windows += s.windows
            continue
        if merged and merged[-1].kind == s.kind and merged[-1].lang == s.lang:
            merged[-1].end = s.end
            merged[-1].windows += s.windows
            continue
        merged.append(s)

    speech = sum(s.dur for s in merged if s.kind == 'speech')
    unknown = sum(s.dur for s in merged if s.kind == 'unknown')
    silence = sum(s.dur for s in merged if s.kind == 'silence')
    music_sec = sum(s.dur for s in merged if s.kind == 'music')
    per: dict[str, float] = {}
    for s in merged:
        if s.kind == 'speech' and s.lang:
            per[s.lang] = per.get(s.lang, 0.0) + s.dur
    props = {k: round(v / speech, 4) for k, v in sorted(per.items(), key=lambda x: -x[1])} \
        if speech else {}
    return Analysis(path=path, duration=dur, window_sec=window, hop_sec=hop,
                    segments=merged, proportions=props, speech_sec=round(speech, 2),
                    unknown_sec=round(unknown, 2), silence_sec=round(silence, 2),
                    music_sec=round(music_sec, 2), lang=lang)


def _ts(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    h, m = divmod(m, 60)
    return f'{h}:{m:02d}:{s:02d}' if h else f'{m}:{s:02d}'


def format_text(a: Analysis) -> str:
    names = LABELS_KO if a.lang == 'ko' else LABELS_EN
    lines = []
    title = '파일' if a.lang == 'ko' else 'file'
    lines.append(f'{title}: {os.path.basename(a.path)}  {_ts(a.duration)} '
                 f'(창 {a.window_sec:g}s · 홉 {a.hop_sec:g}s)')
    lines.append('')
    for s in a.segments:
        if s.kind == 'silence':
            tag = '[무음]' if a.lang == 'ko' else '[silence]'
            lines.append(f'{_ts(s.start):>8}–{_ts(s.end):<8} {tag}')
        elif s.kind == 'unknown':
            tag = '[판정 불가]' if a.lang == 'ko' else '[unknown]'
            lines.append(f'{_ts(s.start):>8}–{_ts(s.end):<8} {tag}')
        elif s.kind == 'music':
            tag = '[음악/효과음]' if a.lang == 'ko' else '[music/sfx]'
            lines.append(f'{_ts(s.start):>8}–{_ts(s.end):<8} {tag}')
        else:
            name = names.get((s.lang or '').split('/')[0], s.lang)
            p = f'{s.mean_p * 100:.0f}%' if s.mean_p is not None else ''
            lines.append(f'{_ts(s.start):>8}–{_ts(s.end):<8} {name:<12} {p:>4}  ({s.windows}창)')
    lines.append('')
    if a.proportions:
        head = '비중(발화 구간만)' if a.lang == 'ko' else 'proportions (speech only)'
        parts = [f'{names.get(k.split("/")[0], k)} {v * 100:.1f}%'
                 for k, v in a.proportions.items()]
        lines.append(f'{head}: ' + ' · '.join(parts))
    total = a.duration or 1.0
    lines.append(('판정 불가' if a.lang == 'ko' else 'unknown') +
                 f' {_ts(a.unknown_sec)} ({a.unknown_sec / total * 100:.1f}%) · ' +
                 ('음악' if a.lang == 'ko' else 'music') +
                 f' {_ts(a.music_sec)} ({a.music_sec / total * 100:.1f}%) · ' +
                 ('무음' if a.lang == 'ko' else 'silence') +
                 f' {_ts(a.silence_sec)} ({a.silence_sec / total * 100:.1f}%)')
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog='eg2_langid.analyze',
                                 description='파일 전체 언어 구간/비중 분석')
    ap.add_argument('file')
    ap.add_argument('--window', type=float, default=5.0)
    ap.add_argument('--hop', type=float, default=2.0)
    ap.add_argument('--min-seg', type=float, default=2.0)
    ap.add_argument('--threads', type=int, default=default_threads(),
                    help='ONNX intra-op 스레드 (기본: 물리 코어 추정치, 최대 6)')
    ap.add_argument('--lanes', type=int, default=None,
                    help='동시 임베딩 레인 (기본 4, 세션 공유)')
    ap.add_argument('--json', dest='json_path')
    ap.add_argument('--lang', default=None, help='출력 언어 ko|en (기본: OS)')
    ap.add_argument('--music', choices=['mark', 'off', 'gate'], default='mark',
                    help='mark(기본): 음악 확률 높아도 언어 margin 높으면 발화 / off: 음악 판정 안 함')
    ap.add_argument('--music-margin', type=float, default=0.3)
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)

    lang = args.lang
    if lang is None:
        try:
            import ctypes
            langid = ctypes.windll.kernel32.GetUserDefaultUILanguage()
            lang = 'ko' if (langid & 0x3FF) == 0x12 else 'en'
        except Exception:
            lang = 'en' if not (os.environ.get('LANG', '').startswith('ko')) else 'ko'

    def prog(t, total):
        if not args.quiet and sys.stderr.isatty():
            sys.stderr.write(f'\r{t:6.1f}/{total:.1f}s ({t / total * 100 if total else 0:3.0f}%)')
            sys.stderr.flush()

    a = analyze_file(args.file, window=args.window, hop=args.hop, min_seg=args.min_seg,
                     threads=args.threads, lang=lang, music=args.music,
                     music_margin=args.music_margin, progress=prog, lanes=args.lanes)
    if not args.quiet and sys.stderr.isatty():
        sys.stderr.write('\r' + ' ' * 40 + '\r')
    print(format_text(a))
    if args.json_path:
        with open(args.json_path, 'w', encoding='utf-8') as f:
            json.dump(a.to_dict(), f, ensure_ascii=False, indent=2)
        print(f'\nJSON: {args.json_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
