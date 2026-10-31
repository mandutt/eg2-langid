"""eg2-langidentifier Phase 1 워커 — FLEURS 언어별 임베딩 (VPS 머신용)

사용:
  python embed_worker.py --langs de_de,es_419 --n 20 --out /data/eg2/out [--cachedir /data/eg2/cache] [--threads 4]

동작:
  1. 언어별 dev.tsv + audio/dev.tar.gz 다운로드 (없으면 — HF resolve URL, 307 redirect 처리)
  2. tar에서 float32 WAV 첫 N개 추출 (WAV format 청크 검사 — FLEURS float32 오독 버그 방지)
  3. gemma2 오디오 임베딩 (q4al4 ONNX, audio encoder + text backbone) → L2 정규화 768d
  4. {lang}_vec.npz (N×768 float32) + {lang}_meta.json (파일명·초) 저장
  5. 전 언어 완료 시 .done 마커 생성 (exec 폴링용)

전제: 패키지 루트에 mmal4/ (ONNX 2종), model-src/tokenizer.json, src/ (open_eg) 존재
"""
import os, sys, io, json, time, wave, struct, tarfile, glob, argparse
import urllib.request
import numpy as np

PKG = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(PKG, 'src')
if os.path.isdir(SRC):
    sys.path.insert(0, SRC)

HF = 'https://huggingface.co/datasets/google/fleurs/resolve/main'
UA = {'User-Agent': 'Mozilla/5.0'}

def fetch(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 10000:
        return dest
    tmp = dest + '.part'
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, 'wb') as f:
        while True:
            b = r.read(1 << 16)
            if not b:
                break
            f.write(b)
    os.replace(tmp, dest)
    return dest

def load_fleurs_wavs(lang, cachedir, n, split='dev'):
    """{split}.tsv + {split}.tar.gz 다운로드 → float32 WAV N개 추출"""
    tsv = fetch(f'{HF}/data/{lang}/{split}.tsv', os.path.join(cachedir, f'{lang}_{split}.tsv'))
    tarp = fetch(f'{HF}/data/{lang}/audio/{split}.tar.gz', os.path.join(cachedir, f'{lang}_{split}.tar.gz'))
    rows = [l.split('\t') for l in open(tsv, encoding='utf-8') if l.strip()]
    wavs, used, srs = [], [], []
    with tarfile.open(tarp, 'r:gz') as tf:
        for r in rows:
            fn = r[1] if len(r) > 1 else r[0]
            if not fn.endswith('.wav'):
                continue
            m = next((x for x in tf.getmembers() if x.name.endswith(fn)), None)
            if m is None:
                continue
            raw = tf.extractfile(m).read()
            if len(raw) < 16000 * 2 * 3:
                continue
            fmt_tag = struct.unpack('<H', raw[20:22])[0]
            sr = struct.unpack('<I', raw[24:28])[0]
            pos, a = 12, None
            while pos < len(raw) - 8:
                csz = struct.unpack('<I', raw[pos+4:pos+8])[0]
                if raw[pos:pos+4] == b'data':
                    dsz = csz - (csz % 4)
                    avail = len(raw) - (pos + 8)
                    if dsz > avail:
                        dsz = avail - (avail % 4)   # 헤더 오기입 wav — 실제 가용 크기로 절축 (10-29: FLEURS fmt3 오기입 실측)
                    if dsz <= 0:
                        continue
                    if fmt_tag == 3:
                        a = np.frombuffer(raw[pos+8:pos+8+dsz], dtype='<f4')
                    else:
                        a = np.frombuffer(raw[pos+8:pos+8+dsz], dtype='<i2').astype(np.float32) / 32768.0
                    break
                pos += 8 + csz + (csz % 2)
            if a is None or len(a) < 16000 * 3:
                continue
            wavs.append(a.astype(np.float32)); used.append(fn); srs.append(sr)
            if len(wavs) >= n:
                break
    return wavs, used, srs

def build_model(pkg, threads):
    from open_eg.processor import preprocess_audio, build_multimodal_input_ids, num_audio_soft_tokens  # noqa: F401
    import torch
    import onnxruntime as ort
    from tokenizers import Tokenizer
    mm = os.path.join(pkg, 'mmal4')
    tok = Tokenizer.from_file(os.path.join(pkg, 'model-src', 'tokenizer.json'))
    tok.enable_truncation(max_length=512)
    so = ort.SessionOptions(); so.intra_op_num_threads = threads
    aenc = ort.InferenceSession(os.path.join(mm, 'audio_encoder_quantized_al4.onnx'), sess_options=so, providers=['CPUExecutionProvider'])
    main_o = ort.InferenceSession(os.path.join(mm, 'model_quantized_al4.onnx'), sess_options=so, providers=['CPUExecutionProvider'])
    empty = np.zeros((0, 512), dtype=np.float32)
    pre = preprocess_audio
    bmid = build_multimodal_input_ids
    nsoft = num_audio_soft_tokens
    return tok, aenc, main_o, empty, pre, bmid, nsoft

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--langs', required=True)
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--out', default='.')
    ap.add_argument('--cachedir', default='/data/eg2/cache')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--startidx', type=int, default=0)
    ap.add_argument('--split', default='dev')
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.cachedir, exist_ok=True)
    langs = [l.strip() for l in a.langs.split(',') if l.strip()]

    tok, aenc, main_o, empty, pre, bmid, nsoft = build_model(os.path.dirname(os.path.abspath(__file__)), a.threads)

    def audio_embed(tt):
        feats, mask = pre([torch.from_numpy(tt)])
        af = aenc.run(['audio_features'], {'input_features': feats.numpy(), 'input_features_mask': mask.numpy()})[0]
        n_soft = nsoft(mask[0])
        ids = bmid([('audio', n_soft)], lambda s: tok.encode(s, add_special_tokens=False))
        v = main_o.run(['sentence_embedding'],
                       {'input_ids': np.array([ids], dtype=np.int64),
                        'attention_mask': np.ones((1, len(ids)), dtype=np.int64),
                        'image_features': empty, 'video_features': empty, 'audio_features': af})[0]
        return v[0] / (np.linalg.norm(v[0]) + 1e-12)

    log = []
    for lang in langs:
        t0 = time.time()
        wavs, used, srs = load_fleurs_wavs(lang, a.cachedir, a.n + a.startidx, a.split)
        wavs = wavs[a.startidx:a.startidx + a.n]
        used = used[a.startidx:a.startidx + a.n]
        if not wavs:
            log.append(f'{lang}: NO_WAVS'); print(f'{lang}: NO_WAVS', flush=True); continue
        vecs = np.array([audio_embed(w) for w in wavs], dtype=np.float32)
        np.savez(os.path.join(a.out, f'{lang}_vec.npz'), vecs=vecs)
        meta = [{'file': f, 'sec': round(len(w) / sr, 2)} for f, w, sr in zip(used, wavs, srs)]
        json.dump(meta, open(os.path.join(a.out, f'{lang}_meta.json'), 'w', encoding='utf-8'), ensure_ascii=False)
        log.append(f'{lang}: {len(wavs)}건 {time.time()-t0:.0f}s')
        print(f'{lang}: {len(wavs)}건 {time.time()-t0:.0f}s', flush=True)

    open(os.path.join(a.out, '.done'), 'w').write('\n'.join(log) + '\n')
    print('DONE', flush=True)

if __name__ == '__main__':
    import torch  # noqa: F401 — build_model 안에서 import하나 토크나이저 로드 순서상 선행
    main()