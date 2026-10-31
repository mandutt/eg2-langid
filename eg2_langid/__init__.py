"""eg2-langid — EmbeddingGemma2 오디오 임베딩 기반 언어 판정 (모델 미포함 경량 패키지).

임베딩 모델은 포함하지 않는다. 768d 벡터를 직접 넘기거나(vector-in),
`detect_audio(pcm, embed_fn=...)`로 임베딩 함수를 주입한다.

    from eg2_langid import LanguageIdentifier
    li = LanguageIdentifier()                 # 아티팩트 ~184KB 로드 (numpy만)
    r = li.detect(vec)                        # 768d → Result
    r = li.detect_audio(pcm, embed_fn=embed)  # 오디오 → Result

선택적 어댑터(모델 로더)는 `eg2_langid.adapters.gemma2` (import 시에만 onnxruntime 사용).
"""
from .detector import LanguageIdentifier, Result

__all__ = ['LanguageIdentifier', 'Result']
__version__ = '1.4.0'
