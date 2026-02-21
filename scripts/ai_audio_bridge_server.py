#!/usr/bin/env python3
"""
Realtime/offline audio bridge server for LMMS AIAudioBridge plugin.

Realtime protocol (little-endian):
  Header: 4s magic("AIBR"), uint32 frames, uint32 channels, float32 drive
  Payload: frames * channels float32 interleaved audio
Response uses the same header+payload structure.

Offline mode:
  python scripts/ai_audio_bridge_server.py --offline-input in.wav --offline-output out.wav \
    --backend gemini --prompt "clean loud pop mix"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import urllib.request
import wave


MAGIC = b"AIBR"
HEADER = struct.Struct("<4sII f")


def process_block_local(samples: list[float], drive: float) -> list[float]:
    gain = max(0.1, drive)
    return [math.tanh(s * gain) for s in samples]


def process_block_cloud_placeholder(samples: list[float], drive: float, backend: str) -> list[float]:
    # Realtime cloud round-trips are not practical for live playback latency.
    # We intentionally keep processing local for hard realtime while still allowing
    # backend/key wiring for future offline or hybrid modes.
    _ = backend
    return process_block_local(samples, drive)


def get_cloud_drive_hint(backend: str, prompt: str) -> float:
    prompt = prompt.strip() or "balanced modern mix"
    if backend == "gemini":
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            return 1.2
        body = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "Return ONLY a number between 0.6 and 2.5 for mix drive. "
                                f"Prompt: {prompt}"
                            )
                        }
                    ]
                }
            ]
        }
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            parsed = json.loads(resp.read().decode("utf-8", "ignore"))
        text = (
            parsed.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "1.2")
            .strip()
        )
        try:
            return min(2.5, max(0.6, float(text)))
        except ValueError:
            return 1.2
    if backend == "deepseek":
        key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not key:
            return 1.2
        body = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "Return only a number between 0.6 and 2.5."},
                {"role": "user", "content": f"Choose mix drive for: {prompt}"},
            ],
            "temperature": 0.1,
        }
        req = urllib.request.Request(
            "https://api.deepseek.com/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            parsed = json.loads(resp.read().decode("utf-8", "ignore"))
        text = (
            parsed.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "1.2")
            .strip()
        )
        try:
            return min(2.5, max(0.6, float(text)))
        except ValueError:
            return 1.2
    return 1.2


def run_offline_mode(args: argparse.Namespace) -> int:
    with wave.open(args.offline_input, "rb") as r:
        channels = r.getnchannels()
        sample_width = r.getsampwidth()
        rate = r.getframerate()
        frames = r.getnframes()
        raw = r.readframes(frames)

    if sample_width != 2:
        print("offline mode currently expects 16-bit PCM wav", file=sys.stderr)
        return 5

    count = len(raw) // 2
    ints = struct.unpack("<" + "h" * count, raw)
    samples = [max(-1.0, min(1.0, x / 32768.0)) for x in ints]

    drive = args.drive
    if args.backend in ("gemini", "deepseek"):
        try:
            drive = get_cloud_drive_hint(args.backend, args.prompt)
        except Exception as exc:
            print(f"warning: cloud hint failed ({exc}); using local drive", file=sys.stderr)

    out = process_block_local(samples, drive)
    ints_out = [max(-32768, min(32767, int(x * 32767.0))) for x in out]
    raw_out = struct.pack("<" + "h" * len(ints_out), *ints_out)

    with wave.open(args.offline_output, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_width)
        w.setframerate(rate)
        w.writeframes(raw_out)

    print(
        json.dumps(
            {
                "offline_output": args.offline_output,
                "backend": args.backend,
                "applied_drive": drive,
                "frames": frames,
                "channels": channels,
                "sample_rate": rate,
            }
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["local", "gemini", "deepseek"], default="local")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--offline-input")
    parser.add_argument("--offline-output")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--drive", type=float, default=1.2)
    args = parser.parse_args()

    if args.offline_input or args.offline_output:
        if not args.offline_input or not args.offline_output:
            print("offline mode requires both --offline-input and --offline-output", file=sys.stderr)
            return 6
        return run_offline_mode(args)

    if args.backend == "gemini" and not os.getenv("GEMINI_API_KEY"):
        print("warning: GEMINI_API_KEY not set; using local realtime fallback", file=sys.stderr)
    if args.backend == "deepseek" and not os.getenv("DEEPSEEK_API_KEY"):
        print("warning: DEEPSEEK_API_KEY not set; using local realtime fallback", file=sys.stderr)

    fin = sys.stdin.buffer
    fout = sys.stdout.buffer

    while True:
        header_raw = fin.read(HEADER.size)
        if not header_raw:
            return 0
        if len(header_raw) != HEADER.size:
            return 1

        magic, frames, channels, drive = HEADER.unpack(header_raw)
        if magic != MAGIC:
            return 2
        if channels <= 0 or channels > 8 or frames == 0:
            return 3

        payload_count = frames * channels
        payload_bytes = payload_count * 4
        payload_raw = fin.read(payload_bytes)
        if len(payload_raw) != payload_bytes:
            return 4

        samples = list(struct.unpack("<" + "f" * payload_count, payload_raw))
        if args.backend == "local":
            out = process_block_local(samples, drive)
        else:
            out = process_block_cloud_placeholder(samples, drive, args.backend)

        fout.write(HEADER.pack(MAGIC, frames, channels, drive))
        fout.write(struct.pack("<" + "f" * payload_count, *out))
        fout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
