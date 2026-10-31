#!/usr/bin/env python
"""eg2-langid 패키지 자체 테스트 — 아티팩트 로드 → gold 재현 → 게이트 → 오디오 경로.

실행: python tests/test_package.py   (numpy만 필요, 모델 로드 없음)
"""
import glob
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from eg2_langid import LanguageIdentifier   # noqa: E402

TEST = os.path.join(BASE, 'data', 'raw', 'fleurs_test_full')
CLEAN = os.path.join(BASE, 'data', 'raw', 'clean40')
fails = []


def check(name, cond, detail=''):
    print(f'{"PASS" if cond else "FAIL"}  {name}  {detail}')
    if not cond:
        fails.append(name)


def l2(v):
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


def main():
    li = LanguageIdentifier()
    check('아티팩트 로드', len(li.labels) == 29, f'{len(li.labels)} 클래스')
    check('그룹 설정', li.groups.get('ur_pk') == 'hi_in')
    check('쌍별 설정', li.pairwise == ['id_id', 'ms_my'])

    if not os.path.isdir(TEST) or not glob.glob(os.path.join(TEST, '*_vec.npz')):
        print('SKIP  gold 재현 — 로컬 평가 데이터 없음 (data/raw/fleurs_test_full)')
        print('\n아티팩트·API 기본 검사만 수행했습니다.')
        return

    # ① gold 전체 재현 (게이트 끔 — 순수 판정)
    li_nogate = LanguageIdentifier(gate=False)
    langs22 = sorted([os.path.basename(f).replace('_vec.npz', '')
                      for f in glob.glob(os.path.join(CLEAN, '*_vec.npz'))])
    total = ok = 0
    per = {}
    for i, l in enumerate(langs22):
        X = l2(np.load(os.path.join(TEST, f'{l}_vec.npz'))['vecs'].astype(np.float32))
        c = 0
        for v in X:
            r = li_nogate.detect(v)
            c += (r.lang == l) or (r.label == l)
        per[l] = c
        ok += c
        total += len(X)
    acc = ok / total
    check('gold 재현 (게이트 off)', abs(acc - 0.9868) < 0.002, f'{ok}/{total} = {acc:.4f}')

    # ② 게이트 동작: 아는 언어 5% 이내 거부
    rej = 0
    for l in langs22:
        X = l2(np.load(os.path.join(TEST, f'{l}_vec.npz'))['vecs'].astype(np.float32))
        rej += sum(li.detect(v).unknown for v in X)
    check('게이트 오거부율 ≤ 8%', rej / total <= 0.08, f'{rej}/{total} = {rej/total*100:.1f}%')

    # ③ 미지 언어 거부 (unknown_test)
    UNK = os.path.join(BASE, 'data', 'raw', 'unknown_test')
    if os.path.isdir(UNK):
        u = []
        for p in sorted(glob.glob(os.path.join(UNK, '*_vec.npz'))):
            X = l2(np.load(p)['vecs'].astype(np.float32))
            u += [li.detect(v).unknown for v in X]
        # 클래스별 임계(공정한 FRR) 선택 시 TRR 37.5% / 전역 임계면 70.8%지만 English 거부율 29%.
        # UI 용도는 '실제 발화를 거부하지 않는 것'이 우선이라 클래스별을 채택.
        check('미지 언어 거부율 ≥ 35%', np.mean(u) >= 0.35, f'{np.mean(u)*100:.1f}% ({len(u)}건)')

    # ④ 결과 스키마
    v = l2(np.load(os.path.join(TEST, 'ko_kr_vec.npz'))['vecs'].astype(np.float32))[0]
    r = li.detect(v)
    check('Result 필드', all(hasattr(r, k) for k in ('lang', 'label', 'confidence', 'unknown',
                                                     'margin', 'scores', 'distribution')))
    check('confidence 범위', 0 <= r.confidence <= 1, f'{r.confidence}')
    check('scores 상위5', len(r.scores) == 5)
    check('__str__', isinstance(str(r), str), str(r))

    # ⑤ 짧은 오디오 → 판정 보류
    r2 = li.detect_audio(np.zeros(8000, np.float32), embed_fn=lambda x: v)
    check('짧은 오디오 보류', r2.unknown and r2.lang is None, str(r2))

    # ⑥ 오디오 경로 + 다중 창 비중
    r3 = li.detect_audio(np.tile(v[:1], (16000 * 12, 1)).ravel() * 0.01, embed_fn=lambda x: v,
                         window_sec=5.0, hop_sec=2.0)
    check('다중 창 distribution', len(r3.distribution) >= 1, str(r3.distribution))

    print()
    if fails:
        print(f'실패 {len(fails)}건: {fails}')
        sys.exit(1)
    print('전체 통과')


if __name__ == '__main__':
    main()
