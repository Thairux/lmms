# AI Audio Bridge

`AIAudioBridge` is a native LMMS effect plugin that adds a realtime AI-ready audio bridge path.

## Transport modes

- `Pipe`: sends each block to an external process (`scripts/ai_audio_bridge_server.py`) over stdin/stdout.
- `Shared Memory`: runs a low-latency local shaping path inside LMMS worker memory (no external process, no API).

## Backends

- `Local`: always free and realtime-safe.
- `Gemini`: key wiring supported via `GEMINI_API_KEY` or plugin API key field.
- `DeepSeek`: key wiring supported via `DEEPSEEK_API_KEY` or plugin API key field.

For realtime playback, cloud modes currently use local processing fallback to avoid network latency glitches.

## External bridge server

Default command:

- `Command`: `python`
- `Script`: `scripts/ai_audio_bridge_server.py`

The script uses a binary frame protocol:

- Header: `"AIBR"`, `frames`, `channels`, `drive`
- Payload: interleaved float32 samples

## Offline Cloud-Assisted Render

You can use cloud backends out of realtime with the same bridge script:

```bash
python scripts/ai_audio_bridge_server.py \
  --offline-input in.wav \
  --offline-output out.wav \
  --backend gemini \
  --prompt "loud clean modern mix"
```

For DeepSeek replace `--backend gemini` with `--backend deepseek`.

Env keys:

- `GEMINI_API_KEY`
- `DEEPSEEK_API_KEY`
