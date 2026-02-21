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
import hashlib
import json
import math
import os
import random
import re
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


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def parse_bpm(prompt: str, fallback: int) -> int:
    m = re.search(r"(\d{2,3})\s*bpm", prompt.lower())
    if not m:
        return fallback
    bpm = int(m.group(1))
    return max(60, min(200, bpm))


def midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def pick_seed(prompt: str) -> int:
    raw = hashlib.sha256(prompt.encode("utf-8", "ignore")).digest()
    return int.from_bytes(raw[:8], "little", signed=False)


def run_generate_mode(args: argparse.Namespace) -> int:
    prompt = args.prompt.strip() or "energetic electronic track"
    seed = pick_seed(prompt)
    rng = random.Random(seed)

    sr = int(args.sample_rate)
    channels = max(1, min(2, int(args.channels)))
    seconds = max(4, min(300, int(args.generate_seconds)))
    bpm = parse_bpm(prompt, int(args.generate_bpm) if args.generate_bpm else rng.randint(110, 150))
    beat_s = 60.0 / bpm
    n = int(seconds * sr)

    left = [0.0] * n
    right = [0.0] * n

    base_midi = rng.choice([45, 48, 50, 52])  # A2, C3, D3, E3
    progression = rng.choice(
        [
            [0, 5, 3, 4],
            [0, 3, 5, 4],
            [0, 7, 5, 3],
            [0, 4, 5, 3],
        ]
    )

    # Kick (quarter note)
    kick_len = int(0.22 * sr)
    for beat in range(int(seconds / beat_s) + 1):
        start = int(beat * beat_s * sr)
        for j in range(kick_len):
            i = start + j
            if i >= n:
                break
            t = j / sr
            env = math.exp(-t * 20.0)
            freq = 140.0 - 100.0 * min(1.0, t * 8.0)
            s = math.sin(2.0 * math.pi * freq * t) * env * 0.9
            left[i] += s
            right[i] += s

    # Snare (beats 2 and 4)
    snare_len = int(0.18 * sr)
    for bar in range(int(seconds / (beat_s * 4)) + 1):
        for off in (1, 3):
            start = int((bar * 4 + off) * beat_s * sr)
            for j in range(snare_len):
                i = start + j
                if i >= n:
                    break
                t = j / sr
                env = math.exp(-t * 30.0)
                noise = (rng.random() * 2.0 - 1.0) * env * 0.4
                tone = math.sin(2.0 * math.pi * 190.0 * t) * env * 0.2
                s = noise + tone
                left[i] += s
                right[i] += s

    # Hi-hat (1/8 notes)
    hat_len = int(0.04 * sr)
    step = beat_s / 2.0
    for k in range(int(seconds / step) + 1):
        start = int(k * step * sr)
        pan = -0.3 if (k % 2 == 0) else 0.3
        for j in range(hat_len):
            i = start + j
            if i >= n:
                break
            t = j / sr
            env = math.exp(-t * 80.0)
            noise = (rng.random() * 2.0 - 1.0) * env * 0.15
            left[i] += noise * (1.0 - pan * 0.5)
            right[i] += noise * (1.0 + pan * 0.5)

    # Bass + lead
    sixteenth = beat_s / 4.0
    for step_idx in range(int(seconds / sixteenth) + 1):
        t0 = step_idx * sixteenth
        i0 = int(t0 * sr)
        if i0 >= n:
            break
        bar = int(t0 / (beat_s * 4.0))
        prog = progression[bar % len(progression)]
        root = base_midi + prog
        bass_hz = midi_to_hz(root - 12)
        lead_hz = midi_to_hz(root + rng.choice([12, 15, 19]))
        gate = 1.0 if (step_idx % 2 == 0) else 0.6
        seg_len = int(sixteenth * sr)
        for j in range(seg_len):
            i = i0 + j
            if i >= n:
                break
            t = j / sr
            phase_b = 2.0 * math.pi * bass_hz * (t0 + t)
            bass = (math.sin(phase_b) * 0.6 + math.sin(phase_b * 0.5) * 0.4) * 0.28 * gate
            phase_l = 2.0 * math.pi * lead_hz * (t0 + t)
            lead = (math.sin(phase_l) + math.sin(phase_l * 1.01)) * 0.08 * (0.5 + 0.5 * math.sin(2 * math.pi * 0.2 * (t0 + t)))
            left[i] += bass + lead * 0.8
            right[i] += bass + lead * 1.2

    # Light sidechain pump by kick envelope
    for beat in range(int(seconds / beat_s) + 1):
        start = int(beat * beat_s * sr)
        pump_len = int(0.22 * sr)
        for j in range(pump_len):
            i = start + j
            if i >= n:
                break
            amt = 0.35 * math.exp(-j / (0.08 * sr))
            left[i] *= (1.0 - amt)
            right[i] *= (1.0 - amt)

    interleaved: list[float] = []
    peak = 1e-9
    for i in range(n):
        peak = max(peak, abs(left[i]), abs(right[i]))
    norm = 0.9 / peak

    drive = 1.1
    if args.backend in ("gemini", "deepseek"):
        try:
            drive = get_cloud_drive_hint(args.backend, prompt)
        except Exception as exc:
            print(f"warning: cloud hint failed ({exc}); using local generation drive", file=sys.stderr)

    for i in range(n):
        l = clamp(left[i] * norm, -1.0, 1.0)
        r = clamp(right[i] * norm, -1.0, 1.0)
        l = math.tanh(l * drive)
        r = math.tanh(r * drive)
        if channels == 1:
            interleaved.append((l + r) * 0.5)
        else:
            interleaved.extend([l, r])

    ints_out = [max(-32768, min(32767, int(x * 32767.0))) for x in interleaved]
    raw_out = struct.pack("<" + "h" * len(ints_out), *ints_out)

    with wave.open(args.generate_output, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(raw_out)

    print(
        json.dumps(
            {
                "generate_output": args.generate_output,
                "prompt": prompt,
                "backend": args.backend,
                "seconds": seconds,
                "bpm": bpm,
                "sample_rate": sr,
                "channels": channels,
                "applied_drive": drive,
            }
        )
    )
    return 0


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
    parser.add_argument("--generate-output")
    parser.add_argument("--generate-seconds", type=int, default=24)
    parser.add_argument("--generate-bpm", type=int)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--drive", type=float, default=1.2)
    args = parser.parse_args()

    if args.generate_output:
        return run_generate_mode(args)

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
