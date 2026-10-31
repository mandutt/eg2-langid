# Bundled assets

## silero_vad.onnx (2.2 MB)

Voice-activity-detection model used by `eg2_langid.nonspeech` to reach 97.5 % music detection at
0 % speech false-positives (the numpy-only path manages 71.6 %).

| | |
|---|---|
| Source | https://github.com/snakers4/silero-vad — `src/silero_vad/data/silero_vad.onnx` |
| License | **MIT**, Copyright (c) 2020-present Silero Team — full text in `SILERO_VAD_LICENSE.txt` |
| Redistributed | unchanged (byte-identical to the upstream file) |
| Size | 2,327,524 bytes |
| sha256 | `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3` |

The upstream README states: *"Published under permissive license (MIT) Silero VAD has zero strings
attached - no telemetry, no keys, no registration, no built-in expiration, no keys or vendor lock."*

The adapter (`eg2_langid/adapters/silero_vad.py`) also accepts the `h`/`c`-split v6 build via
`EG2_VAD_MODEL`, but the bundled file is the upstream state-layout model so provenance stays simple.
