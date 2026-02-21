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
- `MusicGen`

For realtime playback, cloud backends currently use a local processing fallback to avoid unstable latency.

## Generate Songs From Scratch

Use the bridge in generation mode:

```bash
python scripts/ai_audio_bridge_server.py \
  --generate-output scripts/out.wav \
  --prompt "UK drill beat, dark 808 slides, sparse piano, 142 bpm" \
  --backend musicgen \
  --engine musicgen \
  --musicgen-model facebook/musicgen-small \
  --generate-seconds 180 \
  --auto-learn
```

If `torch/transformers` are unavailable, it falls back to the built-in synth engine.
On this machine, `torch` for Python 3.14 may fail to load (`shm.dll` error). For real MusicGen generation, use Python 3.11 or 3.12.

## Self-Learning Loop

1. Generate track(s).
2. Rate each result:

```bash
pwsh scripts/rate_song.ps1 -TrackPath scripts/out.wav -Prompt "UK drill beat" -Rating 5 -Style uk_drill -Bpm 142
```

3. Rebuild profile anytime:

```bash
pwsh scripts/train_ai_profile.ps1
```

The profile is saved at `data/ai_training/profile.json` and automatically applied in `--auto-learn` mode.

## Build Your Own Labeled Library

```bash
python scripts/build_labeled_library.py --input-dir path/to/your_loops --output data/ai_training/labeled_prompts.jsonl
```

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

## Data Library

Starter labeled library is included in:

- `data/ai_training/labeled_prompts.jsonl`
