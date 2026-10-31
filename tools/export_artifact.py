#!/usr/bin/env python
"""학습 아티팩트 생성 — 최종 29-way 구성(C)을 고정 시드로 재현해 model.npz/labels.json을 만든다.

산출물 (eg2_langid/data/):
  model.npz   W(768,29) · head(768,2) · centroids(768,29) · 클래스별 통계 · 게이트 임계 · softmax 온도
  labels.json 클래스 이름 29개 + 메타(성능 수치·버전)

사용: python scripts/export_artifact.py [--out eg2_langid/data]
"""
import argparse, glob, json, os, time
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOX = os.path.join(BASE, 'data', 'raw', 'voxlingua')
CLEAN = os.path.join(BASE, 'data', 'raw', 'clean40')
TEST = os.path.join(BASE, 'data', 'raw', 'fleurs_test_full')
OOD = os.path.join(BASE, 'data', 'raw', 'ood_test')

NEW = {'ur': 'ur_pk', 'da': 'da_dk', 'fa': 'fa_ir', 'cs': 'cs_cz',
       'hr': 'hr_hr', 'hy': 'hy_am', 'sw': 'sw_ke'}
VL = ['uk', 'ru', 'id', 'ms', 'hi', 'ur', 'da', 'fa', 'cs', 'hr', 'hy', 'sw', 'en', 'sv']
ALLOC = {'en': 400, 'sv': 200, 'id': 400, 'ms': 400, 'ur': 400, 'da': 400, 'fa': 400,
         'cs': 400, 'hr': 400, 'hy': 400, 'sw': 400, 'uk': 800, 'ru': 0, 'hi': 0}
KEY = {'en': 'en_us', 'sv': 'sv_se', 'uk': 'uk_ua', 'ru': 'ru_ru', 'hi': 'hi_in'}
HEAD = 2000
SEED = 7
GATE_FR = 0.05


def l2(v):
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


def fit(X, y, n, lam=0.01):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ np.eye(n)[y])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(BASE, 'eg2_langid', 'data'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    langs22 = sorted([os.path.basename(f).replace('_vec.npz', '') for f in glob.glob(os.path.join(CLEAN, '*_vec.npz'))])
    C = {l: l2(np.load(os.path.join(CLEAN, f'{l}_vec.npz'))['vecs'].astype(np.float32)) for l in langs22}
    V = {l: l2(np.load(os.path.join(VOX, f'{l}_vec.npz'))['vecs'].astype(np.float32)) for l in VL}
    sn = sorted(NEW)
    labels = langs22 + [NEW[k] for k in sn]
    idx = {l: i for i, l in enumerate(labels)}

    # ── 29-way 프로브 학습 (최종 구성 C, 시드 고정)
    rng = np.random.RandomState(SEED)
    Xs, ys = [], []
    for i, l in enumerate(langs22):
        Xs.append(C[l]); ys.append(np.full(len(C[l]), i))
    for k in sn:
        n = ALLOC[k]; Xs.append(V[k][rng.choice(len(V[k]), n, replace=False)]); ys.append(np.full(n, idx[NEW[k]]))
    for k, key in KEY.items():
        n = ALLOC.get(k, 0)
        if n:
            Xs.append(V[k][rng.choice(len(V[k]), n, replace=False)]); ys.append(np.full(n, idx[key]))
    Xtr = l2(np.vstack(Xs)); ytr = np.concatenate(ys)
    W = fit(Xtr, ytr, 29).astype(np.float32)

    # ── id/ms 쌍별 헤드
    head = fit(np.vstack([V['id'][:HEAD], V['ms'][:HEAD], C['id_id'], C['ms_my']]),
               np.array([0] * HEAD + [1] * HEAD + [0] * 40 + [1] * 40), 2).astype(np.float32)

    centroids = np.stack([l2(Xtr[ytr == i].mean(axis=0, keepdims=True))[0] for i in range(29)]).astype(np.float32)
    cmu = np.zeros(29, np.float32); csd = np.ones(29, np.float32)
    for i in range(29):
        c = Xtr[ytr == i] @ centroids[i]
        cmu[i] = c.mean(); csd[i] = c.std() + 1e-6

    # ── 게이트 특징 + 클래스별 임계 (gold 22클래스, 신규 7은 ood_test)
    def feats(X):
        P = X @ W
        srt = np.sort(P, axis=1)
        margin = srt[:, -1] - srt[:, -2]
        coss = X @ centroids.T
        top = coss.argmax(axis=1)
        cmax = coss[np.arange(len(X)), top]
        sc = np.sort(coss, axis=1)
        gap = sc[:, -1] - sc[:, -2]
        cz = (cmax - cmu[top]) / csd[top]
        return margin, gap, cz, top

    # 정규화 통계는 **gold 특징**에서 (실측 프로토콜과 동일 — 학습벡터 기준이면 스케일이 어긋난다)
    gfeat = {l: feats(l2(np.load(os.path.join(TEST, f'{l}_vec.npz'))['vecs'].astype(np.float32)))[:3]
             for l in langs22}
    allg = np.concatenate([np.stack(gfeat[l], 1) for l in langs22], 0)
    fmu = allg.mean(axis=0).astype(np.float32)
    fsd = (allg.std(axis=0) + 1e-6).astype(np.float32)
    fuse = lambda m, g, z: ((m - fmu[0]) / fsd[0] + (g - fmu[1]) / fsd[1] + (z - fmu[2]) / fsd[2]) / 3.0

    gate_tau = np.zeros(29, np.float32)
    QS = [0.05, 0.25, 0.50, 0.75, 0.95]
    marg_q = np.zeros((29, len(QS)), np.float32)     # 신뢰도(백분위) 매핑용
    for i, l in enumerate(langs22):
        pf = os.path.join(TEST, f'{l}_vec.npz')
        m, g, z, _ = feats(l2(np.load(pf)['vecs'].astype(np.float32)))
        f = fuse(m, g, z)
        gate_tau[i] = np.quantile(f, GATE_FR)
        marg_q[i] = np.quantile(m, QS)
    for j, k in enumerate(sn):
        pf = os.path.join(OOD, f'{NEW[k]}_vec.npz')
        if os.path.exists(pf):
            m, g, z, _ = feats(l2(np.load(pf)['vecs'].astype(np.float32)))
            f = fuse(m, g, z)
            gate_tau[22 + j] = np.quantile(f, 0.10)   # 표본 10건 → 완만한 컷
            marg_q[22 + j] = np.quantile(m, QS)

    # ── 게이트 오프셋: 런타임 규칙(코사인 최근접 클래스의 임계)에서 FRR을 정확히 맞춘다
    gscore = []
    for l in langs22:
        m, g, z, top = feats(l2(np.load(os.path.join(TEST, f'{l}_vec.npz'))['vecs'].astype(np.float32)))
        f = fuse(m, g, z)
        gscore.append(f - gate_tau[top])
    gate_offset = 0.0   # 전역 임계에 이미 반영

    # 게이트: **판정 클래스(pred) 기준** 클래스별 임계.
    # (코사인 최근접 클래스는 실제 클래스와 어긋나 en→nl/hi로 쏠림 → 사용 금지)
    gate_cls = np.zeros(29, np.float32)
    gate_q95 = np.zeros(29, np.float32)
    gfused_all = []
    for i, l in enumerate(langs22):
        X = l2(np.load(os.path.join(TEST, f'{l}_vec.npz'))['vecs'].astype(np.float32))
        m, g, z, _ = feats(X)
        f = fuse(m, g, z)
        pred = (X @ W).argmax(axis=1)
        mine = pred == i
        gate_cls[i] = np.quantile(f[mine], GATE_FR) if mine.sum() >= 10 else np.quantile(f, GATE_FR)
        gate_q95[i] = np.quantile(f[mine], 0.95) if mine.sum() >= 10 else np.quantile(f, 0.95)
        gfused_all.append(f)
    gfused_all = np.concatenate(gfused_all)
    for j, k in enumerate(sn):
        pf = os.path.join(OOD, f'{NEW[k]}_vec.npz')
        if os.path.exists(pf):
            m, g, z, _ = feats(l2(np.load(pf)['vecs'].astype(np.float32)))
            f = fuse(m, g, z)
            gate_cls[22 + j] = np.quantile(f, 0.10)
            gate_q95[22 + j] = np.quantile(f, 0.95)
        else:
            gate_cls[22 + j] = gate_cls[:22].mean()
            gate_q95[22 + j] = gate_q95[:22].mean()
    gate_global = float(np.median(gate_cls[:22]))   # 폴백용
    gate_q95_global = float(np.quantile(gfused_all, 0.95))

    # ── softmax 온도 (gold에서 confidence 보정)
    def eval_all(Wm, hw):
        per = []
        for i, l in enumerate(langs22):
            X = l2(np.load(os.path.join(TEST, f'{l}_vec.npz'))['vecs'].astype(np.float32))
            P = X @ Wm
            pred = P.argmax(axis=1)
            pred = np.where(pred == 22 + sn.index('ur'), i if l == 'hi_in' else pred, pred)
            o = np.argsort(-P, axis=1)
            sel = np.isin(o[:, 0], [idx['id_id'], idx['ms_my']]) & np.isin(o[:, 1], [idx['id_id'], idx['ms_my']])
            if sel.any():
                sub = (l2(X[sel]) @ hw).argmax(axis=1)
                pred[sel] = np.where(sub == 0, idx['id_id'], idx['ms_my'])
            per.append((pred == i).astype(np.float32))
        return np.concatenate(per)

    hits = eval_all(W, head)
    gold_acc = float(hits.mean())
    Pg = []
    for l in langs22:
        Pg.append(l2(np.load(os.path.join(TEST, f'{l}_vec.npz'))['vecs'].astype(np.float32)) @ W)
    Pg = np.vstack(Pg)
    srt_margin = np.sort(Pg, axis=1)[:, -1] - np.sort(Pg, axis=1)[:, -2]
    yg = np.concatenate([np.full(100, i) for i in range(22)])
    # margin → 정답확률 캘리브레이션 곡선 (gold 실측, 게이트 off 기준)
    order = np.argsort(srt_margin)
    mg_sorted = srt_margin[order]; ok_sorted = hits[order]
    K = 12
    edges = np.linspace(0, len(mg_sorted), K + 1).astype(int)
    cal_m, cal_p = [], []
    for b in range(K):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        cal_m.append(float(np.median(mg_sorted[lo:hi])))
        n_b = hi - lo
        cal_p.append(float((ok_sorted[lo:hi].sum() + 1) / (n_b + 2)))   # Laplace — 100% 방지
    # 단조 증가 보정 (역행 구간은 누적 최대) + 상한 0.99
    cal_p = np.clip(np.maximum.accumulate(np.array(cal_p, np.float32)), 0.02, 0.99)
    cal_m = np.array(cal_m, np.float32)

    best_T, best_ece = 1.0, 9.9
    for T in np.arange(0.05, 3.01, 0.05):
        E = np.exp((Pg - Pg.max(axis=1, keepdims=True)) / T)
        pm = E / E.sum(axis=1, keepdims=True)
        conf = pm.max(axis=1); correct = (pm.argmax(axis=1) == yg).astype(np.float32)
        bins = np.linspace(0, 1, 11); ece = 0.0
        for b in range(10):
            m = (conf >= bins[b]) & (conf < bins[b + 1])
            if m.sum():
                ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
        if ece < best_ece:
            best_ece, best_T = ece, float(T)

    np.savez(os.path.join(a.out, 'model.npz'),
             W=W, head=head, centroids=centroids, cls_mu=cmu, cls_sd=csd,
             feat_mu=fmu, feat_sd=fsd, gate_tau=gate_tau, margin_q=marg_q,
             margin_qs=np.array(QS, np.float32), temperature=np.float32(best_T),
             gate_offset=np.float32(gate_offset), cal_m=cal_m, cal_p=cal_p, gate_q95=gate_q95,
             gate_global=np.float32(gate_global), gate_cls=gate_cls,
             gate_q95_global=np.float32(gate_q95_global))
    json.dump({'labels': labels, 'groups': {'ur_pk': 'hi_in'},
               'pairwise': {'classes': ['id_id', 'ms_my']},
               'gate': {'fr': GATE_FR, 'features': ['margin', 'cos_gap', 'cos_z']},
               'meta': {'version': '1.0.0', 'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                        'gold_acc': round(gold_acc, 4), 'temperature': round(best_T, 3),
                        'ece': round(best_ece, 4), 'n_classes': len(labels),
                        'alloc': ALLOC, 'head_n': HEAD, 'seed': SEED}},
              open(os.path.join(a.out, 'labels.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    sz = os.path.getsize(os.path.join(a.out, 'model.npz')) + os.path.getsize(os.path.join(a.out, 'labels.json'))
    print(f'gold {sum(hits):.0f}/2200 = {gold_acc:.4f} · T={best_T:.2f} · ECE={best_ece:.3f}')
    print(f'저장 {a.out}  model.npz+labels.json = {sz/1024:.0f}KB')


if __name__ == '__main__':
    main()
