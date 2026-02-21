# AIAudioBridge Effect Plugin

`AIAudioBridge` is an effect plugin for realtime AI-ready audio processing.

## Location

- Plugin sources: `plugins/AIAudioBridge/`
- Bridge server: `scripts/ai_audio_bridge_server.py`

## Realtime Modes

`AIAudioBridge` provides two transport modes:

- `Pipe`:
  - Sends each audio block to an external process over stdin/stdout.
  - Default bridge command/script:
    - Command: `python`
    - Script: `scripts/ai_audio_bridge_server.py`
- `Shared Memory`:
  - Uses in-process low-latency processing path.
  - No external process required.

## Backend Modes

- `Local`
- `Gemini`
- `DeepSeek`

For realtime playback, cloud backends currently use a local processing fallback to avoid unstable latency.

## Offline Cloud-Assisted Render

The bridge server supports offline file processing:

```bash
python scripts/ai_audio_bridge_server.py \
  --offline-input in.wav \
  --offline-output out.wav \
  --backend gemini \
  --prompt "loud clean modern mix"
```

DeepSeek example:

```bash
python scripts/ai_audio_bridge_server.py \
  --offline-input in.wav \
  --offline-output out.wav \
  --backend deepseek \
  --prompt "tight low end, brighter top end"
```

### How it works

- The script reads the input WAV.
- For cloud backends, it requests a lightweight drive hint from the provider.
- It applies local DSP to produce output audio (deterministic and fast).
- It writes a processed WAV to `--offline-output`.

## Do You Need to Download a Model?

Short answer: no.

- `Local` mode uses built-in processing in the bridge script.
- `Gemini` and `DeepSeek` modes call hosted APIs and do not require local model files.

## Keys

Use environment variables or plugin fields:

- `GEMINI_API_KEY`
- `DEEPSEEK_API_KEY`
