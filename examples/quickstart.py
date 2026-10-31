"""Minimal usage — judge a precomputed 768-d embedding.

Run: python examples/quickstart.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eg2_langid import LanguageIdentifier


def main() -> None:
    li = LanguageIdentifier()
    print(f"{len(li.labels)} languages loaded, artifact v{li.meta.get('version')}")

    # any L2-normalized 768-d vector (e.g. from an EmbeddingGemma 2 audio encoder)
    vec = np.random.RandomState(0).randn(768).astype(np.float32)
    vec /= np.linalg.norm(vec)

    r = li.detect(vec)
    print(f"result      : {r}")
    print(f"unknown     : {r.unknown}")
    print(f"confidence  : {r.confidence}  (headroom above the rejection gate)")
    print(f"p_correct   : {r.p_correct}   (calibrated on held-out data)")
    print(f"top-5       : {list(r.scores.items())[:5]}")


if __name__ == "__main__":
    main()
