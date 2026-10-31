"""비발화(음악·효과음) 판별 — 신호 특징 + 소형 로지스틱.

임베딩 없이 5초 파형에서 10~11개 특징을 뽑아 "말인가, 음악/효과음인가"를 판정합니다.
numpy 만으로 동작하며, VAD 함수를 주면 정확도가 올라갑니다(있으면 97.5%, 없으면 71.6% 검출).
"""
from __future__ import annotations

import numpy as np

SR = 16000
FRAME = 400          # 25 ms
HOP = 160            # 10 ms

KEYS = ('vad', 'mod2_8', 'zcr', 'centroid_mean', 'centroid_sd', 'flat_mean',
        'flat_sd', 'flux', 'rms_mean', 'rms_sd', 'rms_cv')


def features(pcm: np.ndarray, vad_fn=None) -> np.ndarray:
    """5초 파형(16kHz float32) → 특징 벡터.

    vad_fn: (pcm) -> float in [0,1] 발화 비율. 없으면 해당 특징 생략(10개).
    """
    a = np.asarray(pcm, dtype=np.float32).ravel()
    if len(a) < FRAME * 2:
        a = np.pad(a, (0, FRAME * 2 - len(a)))

    frames = np.lib.stride_tricks.sliding_window_view(a, FRAME)[::HOP]
    win = np.hanning(FRAME)
    S = np.abs(np.fft.rfft(frames * win, axis=1)) + 1e-10
    freqs = np.fft.rfftfreq(FRAME, 1.0 / SR)

    centroid = (S * freqs).sum(1) / S.sum(1)
    flatness = np.exp(np.log(S).mean(1)) / S.mean(1)
    flux = float(np.sqrt((np.diff(S, axis=0) ** 2).mean(1)).mean())
    zcr = float((np.diff(np.sign(a)) != 0).mean())
    rms = np.sqrt((frames ** 2).mean(1))

    env = rms - rms.mean()
    E = np.abs(np.fft.rfft(env))
    fr = np.fft.rfftfreq(len(env), HOP / SR)
    mod = float(E[(fr >= 2) & (fr <= 8)].sum() / (E.sum() + 1e-10))

    vals = [
        float(vad_fn(a)) if vad_fn is not None else None,
        mod, zcr,
        float(centroid.mean()), float(centroid.std()),
        float(flatness.mean()), float(flatness.std()),
        flux,
        float(rms.mean()), float(rms.std()),
        float(rms.std() / (rms.mean() + 1e-9)),
    ]
    return np.array([v for v in vals if v is not None], np.float32)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class NonSpeechDetector:
    """model.npz 의 ns_* 배열로 초기화. 없으면 비활성."""

    def __init__(self, art):
        self.ok = 'ns_novad_w' in art
        if not self.ok:
            return
        # 프로브 반응 특징(하이브리드용)에 필요한 가중치
        self.W = art['W'].astype(np.float32) if 'W' in art else None
        self.centroids = art['centroids'].astype(np.float32) if 'centroids' in art else None
        self.cls_mu = art['cls_mu'].astype(np.float32) if 'cls_mu' in art else None
        self.cls_sd = art['cls_sd'].astype(np.float32) if 'cls_sd' in art else None
        self.sets = {}
        for name in ('vad', 'novad', 'hybrid'):
            if f'ns_{name}_w' not in art:
                continue
            self.sets[name] = dict(
                mu=art[f'ns_{name}_mu'].astype(np.float32),
                sd=art[f'ns_{name}_sd'].astype(np.float32),
                w=art[f'ns_{name}_w'].astype(np.float32),
                b=float(art[f'ns_{name}_b']),
                w3=art[f'ns_{name}_w3'].astype(np.float32) if f'ns_{name}_w3' in art else None,
                b3=art[f'ns_{name}_b3'].astype(np.float32) if f'ns_{name}_b3' in art else None,
                thr=float(art.get(f'ns_{name}_thr', 0.7)),
            )
        self.classes3 = [str(c) for c in art['ns_classes3']] if 'ns_classes3' in art else \
            ['speech', 'music', 'sfx']
        # 가드레일: 창의 이 비율 이상이 발화로 잡히면 음악 판정 금지 (시끄러운 인터뷰·게임 방송 대응)
        self.guard = float(art['ns_guard']) if 'ns_guard' in art else 0.20

    def probe_features(self, vec) -> np.ndarray | None:
        """768d 임베딩 → 프로브 반응 6특징 (최고점수·margin·엔트로피·코사인 최대/간격/z)."""
        if self.W is None or self.centroids is None or vec is None:
            return None
        x = np.asarray(vec, np.float32).ravel()
        x = x / (np.linalg.norm(x) + 1e-12)
        P = x @ self.W
        srt = np.sort(P)
        Z = P / 0.1
        Z = Z - Z.max()
        e = np.exp(Z)
        ent = float(-(e / e.sum() * np.log(e / e.sum() + 1e-12)).sum())
        cos = self.centroids @ x
        cs = np.sort(cos)
        topc = int(cos.argmax())
        cz = (float(cs[-1]) - float(self.cls_mu[topc])) / float(self.cls_sd[topc])
        return np.array([srt[-1], srt[-1] - srt[-2], ent, cs[-1], cs[-1] - cs[-2], cz], np.float32)

    def score(self, pcm: np.ndarray, vad_fn=None, vec=None, hybrid: bool = False
              ) -> tuple[float, str, float]:
        """→ (비발화 확률, 종류('speech'|'music'|'sfx'), 그 종류의 확률)

        hybrid=True 이고 vec(임베딩)이 주어지면 신호 특징 + 프로브 반응 17특징을 사용한다.
        """
        if not self.ok:
            return 0.0, 'speech', 1.0
        # ① 가드레일 — 발화가 충분히 잡히면 음악/효과음 판정을 취소한다
        v = None
        if vad_fn is not None:
            try:
                v = float(vad_fn(pcm))
            except Exception:
                v = None
        if v is not None and v >= self.guard:
            return 0.0, 'speech', 1.0
        vf = (lambda _z: v) if v is not None else None
        if hybrid and vec is not None and 'hybrid' in self.sets:
            pf = self.probe_features(vec)
            if pf is not None:
                key = 'hybrid'
            else:
                hybrid = False
        if not (hybrid and 'hybrid' in self.sets):
            key = 'vad' if (vad_fn is not None and 'vad' in self.sets) else 'novad'
        p = self.sets[key]
        x = features(pcm, vf)
        if key == 'hybrid':
            pf = self.probe_features(vec)
            if pf is None:
                return 0.0, 'speech', 1.0
            x = np.concatenate([x, pf])
        if len(x) != len(p['mu']):          # 특징 수 불일치 방어
            return 0.0, 'speech', 1.0
        z = (x - p['mu']) / p['sd']
        prob = float(_sigmoid(np.array([float(z @ p['w'] + p['b'])]))[0])
        kind, kp = 'speech', 1.0 - prob
        if p['w3'] is not None:
            logits = p['w3'] @ z + p['b3']
            e = np.exp(logits - logits.max())
            pr = e / e.sum()
            idx = int(pr.argmax())
            kind, kp = self.classes3[idx], float(pr[idx])
        return prob, kind, kp
