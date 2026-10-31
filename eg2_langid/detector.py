"""판정기 본체 — numpy만 필요. 모델/ONNX 의존성 없음."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Callable, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, 'data')

SR = 16000
MIN_SEC = 1.0          # 이보다 짧으면 판정 보류


@dataclass
class Result:
    """판정 결과. lang=None이면 판정 보류(unknown)."""
    lang: str | None
    label: str | None            # 표시용 (그룹 표기 반영)
    confidence: float            # 0~1 (클래스 내 margin 백분위)
    unknown: bool
    margin: float
    scores: dict[str, float]     # 상위 5개 클래스 점수
    distribution: dict[str, float] = field(default_factory=dict)
    audio_sec: float = 0.0
    p_correct: float | None = None    # gold 캘리브레이션상 정답 확률

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        if self.unknown:
            return f'unknown (margin {self.margin:.3f})'
        return f'{self.label} ({self.confidence * 100:.0f}%)'


def _l2(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


class LanguageIdentifier:
    """29-way 선형 프로브 + {hi,ur} 그룹 + id/ms 쌍별 헤드 + unknown 게이트."""

    def __init__(self, data_dir: str | None = None, gate: bool = True):
        d = data_dir or _DATA
        art = np.load(os.path.join(d, 'model.npz'))
        meta = json.load(open(os.path.join(d, 'labels.json'), encoding='utf-8'))
        self.W = art['W'].astype(np.float32)
        self.head = art['head'].astype(np.float32)
        self.centroids = art['centroids'].astype(np.float32)
        self.cls_mu = art['cls_mu'].astype(np.float32)
        self.cls_sd = art['cls_sd'].astype(np.float32)
        self.feat_mu = art['feat_mu'].astype(np.float32)
        self.feat_sd = art['feat_sd'].astype(np.float32)
        self.gate_tau = art['gate_tau'].astype(np.float32)
        self.margin_q = art['margin_q'].astype(np.float32)
        self.margin_qs = art['margin_qs'].astype(np.float32)
        self.gate_offset = float(art['gate_offset']) if 'gate_offset' in art else 0.0
        self.gate_q95 = art['gate_q95'].astype(np.float32) if 'gate_q95' in art else None
        self.gate_global = float(art['gate_global']) if 'gate_global' in art else None
        self.gate_cls = art['gate_cls'].astype(np.float32) if 'gate_cls' in art else None
        self.gate_q95_global = float(art['gate_q95_global']) if 'gate_q95_global' in art else None
        self.cal_m = art['cal_m'].astype(np.float32) if 'cal_m' in art else None
        self.cal_p = art['cal_p'].astype(np.float32) if 'cal_p' in art else None
        self.labels: list[str] = meta['labels']
        self.meta = meta.get('meta', {})
        self.groups: dict[str, str] = meta.get('groups', {})
        self.pairwise: list[str] = meta.get('pairwise', {}).get('classes', [])
        self.gate_enabled = gate
        self._idx = {l: i for i, l in enumerate(self.labels)}
        self._group_idx = {self._idx[k]: self._idx[v] for k, v in self.groups.items() if k in self._idx}
        self._pair_idx = [self._idx[c] for c in self.pairwise if c in self._idx]

    # ── 핵심: 벡터 → 결과
    def detect(self, vec: Sequence[float] | np.ndarray, audio_sec: float = 0.0) -> Result:
        x = _l2(np.asarray(vec, dtype=np.float32).reshape(1, -1))
        P = (x @ self.W)[0]
        raw = int(P.argmax())

        pred = self._group_idx.get(raw, raw)          # {hi, ur} 그룹 병합

        # id/ms 쌍별 헤드 (top-2가 모두 해당 클래스일 때만)
        if len(self._pair_idx) == 2:
            order = np.argsort(-P)
            if set(order[:2].tolist()) == set(self._pair_idx):
                h = int((x @ self.head)[0].argmax())
                pred = self._pair_idx[h]

        # 게이트 특징
        coss = self.centroids @ x[0]
        cmax = float(coss.max())
        sc = np.sort(coss)
        gap = float(sc[-1] - sc[-2])
        top_c = int(coss.argmax())
        cos_z = (cmax - float(self.cls_mu[top_c])) / float(self.cls_sd[top_c])
        srt = np.sort(P)
        margin = float(srt[-1] - srt[-2])

        fused = ((margin - self.feat_mu[0]) / self.feat_sd[0]
                 + (gap - self.feat_mu[1]) / self.feat_sd[1]
                 + (cos_z - self.feat_mu[2]) / self.feat_sd[2]) / 3.0
        # 판정 클래스(pred) 기준 임계 — 클래스별 FRR 균등(최악 6%)
        cut_global = (float(self.gate_cls[pred]) if self.gate_cls is not None
                      else float(self.gate_tau[pred]) + self.gate_offset)
        unknown = bool(self.gate_enabled and fused < cut_global)

        # ① p_correct: gold 실측 캘리브레이션(정답 확률)
        p_correct = float(np.interp(margin, self.cal_m, self.cal_p)) if self.cal_m is not None else None
        # ② confidence: 거부 경계 대비 여유 (0=간신히 통과, 1=여유롭게 통과)
        cut = cut_global
        top95 = (float(self.gate_q95[pred]) if self.gate_q95 is not None
                 else (self.gate_q95_global or cut + 0.5))
        conf = float(np.clip((fused - cut) / max(top95 - cut, 1e-6), 0.0, 1.0))

        order = np.argsort(-P)[:5]
        scores = {self.labels[i]: round(float(P[i]), 4) for i in order}
        # 그룹 표기 반영한 표시명
        label = self.labels[pred]
        if pred in self._group_idx.values():
            members = [self.labels[k] for k, v in self._group_idx.items() if v == pred]
            label = '/'.join([label] + members)

        return Result(lang=None if unknown else self.labels[pred], label=label,
                      confidence=round(conf, 3), unknown=unknown, margin=round(margin, 4),
                      scores=scores, audio_sec=round(audio_sec, 2),
                      p_correct=None if p_correct is None else round(p_correct, 3))

    # ── 오디오 → 결과 (임베딩은 주입)
    def detect_audio(self, pcm: Sequence[float] | np.ndarray,
                     embed_fn: Callable[[np.ndarray], np.ndarray],
                     window_sec: float = 5.0, hop_sec: float | None = None,
                     sample_rate: int = SR) -> Result:
        """pcm(float32, mono)에서 최근 window_sec 구간을 임베딩해 판정.
        hop_sec을 주면 여러 창을 판정해 다수결 + 비중을 함께 반환."""
        a = np.asarray(pcm, dtype=np.float32).ravel()
        sec = len(a) / sample_rate
        if sec < MIN_SEC:
            return Result(None, None, 0.0, True, 0.0, {}, {}, round(sec, 2))

        win = int(window_sec * sample_rate)
        if hop_sec is None or sec <= window_sec:
            seg = a[-win:] if len(a) > win else a
            return self.detect(embed_fn(np.ascontiguousarray(seg)), audio_sec=len(seg) / sample_rate)

        hop = int(hop_sec * sample_rate)
        starts = list(range(max(0, len(a) - win), -1, -hop))[::-1]
        results = []
        for s in starts:
            seg = a[s:s + win]
            if len(seg) < win:
                continue
            results.append(self.detect(embed_fn(np.ascontiguousarray(seg)), audio_sec=window_sec))
        if not results:
            return self.detect(embed_fn(np.ascontiguousarray(a[-win:])), audio_sec=window_sec)
        counts: dict[str, float] = {}
        for r in results:
            k = r.label if not r.unknown else 'unknown'
            counts[k] = counts.get(k, 0) + 1
        total = sum(counts.values())
        dist = {k: round(v / total, 3) for k, v in sorted(counts.items(), key=lambda x: -x[1])}
        best = max(results, key=lambda r: (r.confidence, not r.unknown))
        best.distribution = dist
        return best
