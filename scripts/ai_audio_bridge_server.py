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
import datetime as dt
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
from pathlib import Path
from typing import Any


MAGIC = b"AIBR"
HEADER = struct.Struct("<4sII f")
STOPWORDS = {
    "a",
    "an",
    "and",
    "the",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "over",
    "under",
    "your",
    "make",
    "song",
    "track",
    "beat",
    "music",
}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def default_data_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir) if args.data_dir else repo_root / "data" / "ai_training"
    feedback_path = Path(args.feedback_path) if args.feedback_path else data_dir / "feedback.jsonl"
    labeled_path = Path(args.labeled_path) if args.labeled_path else data_dir / "labeled_prompts.jsonl"
    profile_path = Path(args.profile_path) if args.profile_path else data_dir / "profile.json"
    generations_path = data_dir / "generations.jsonl"
    return feedback_path, labeled_path, profile_path, generations_path


def tokenize_prompt(prompt: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]+", prompt.lower())
    return [w for w in words if len(w) >= 3 and w not in STOPWORDS]


def build_learning_profile(feedback_rows: list[dict[str, Any]], labeled_rows: list[dict[str, Any]]) -> dict[str, Any]:
    style_sum: dict[str, float] = {}
    style_weight: dict[str, float] = {}
    bpm_sum = 0.0
    bpm_weight = 0.0
    drive_sum = 0.0
    drive_weight = 0.0
    guidance_sum = 0.0
    guidance_weight = 0.0
    temp_sum = 0.0
    temp_weight = 0.0

    rows = list(labeled_rows) + list(feedback_rows)
    for row in rows:
        prompt = str(row.get("prompt", ""))
        style = str(row.get("style", "") or infer_style(prompt)).strip().lower().replace(" ", "_")
        rating = clamp(safe_float(row.get("rating", row.get("quality", 3.0)), 3.0), 1.0, 5.0)
        weight = 0.25 + (rating / 5.0)
        style_sum[style] = style_sum.get(style, 0.0) + (rating * weight)
        style_weight[style] = style_weight.get(style, 0.0) + weight

        bpm = safe_int(row.get("bpm", 0), 0)
        if bpm > 0:
            bpm_sum += bpm * weight
            bpm_weight += weight
        drive = safe_float(row.get("drive", 0.0), 0.0)
        if drive > 0:
            drive_sum += drive * weight
            drive_weight += weight
        guidance = safe_float(row.get("guidance", 0.0), 0.0)
        if guidance > 0:
            guidance_sum += guidance * weight
            guidance_weight += weight
        temp = safe_float(row.get("temperature", 0.0), 0.0)
        if temp > 0:
            temp_sum += temp * weight
            temp_weight += weight

    style_scores = {k: (style_sum[k] / max(1e-6, style_weight.get(k, 1.0))) for k in style_sum}
    default_style = ""
    if style_scores:
        default_style = sorted(style_scores.items(), key=lambda kv: kv[1], reverse=True)[0][0]

    return {
        "version": 1,
        "updated_at": now_iso(),
        "examples": len(rows),
        "style_scores": style_scores,
        "default_style": default_style,
        "preferred_bpm": int(round(bpm_sum / bpm_weight)) if bpm_weight > 0 else 0,
        "preferred_drive": round(drive_sum / drive_weight, 4) if drive_weight > 0 else 1.2,
        "preferred_guidance": round(guidance_sum / guidance_weight, 4) if guidance_weight > 0 else 3.5,
        "preferred_temperature": round(temp_sum / temp_weight, 4) if temp_weight > 0 else 1.0,
    }


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


def infer_style(prompt: str) -> str:
    p = prompt.lower()
    if "uk drill" in p or ("drill" in p and "uk" in p):
        return "uk_drill"
    if "drill" in p:
        return "drill"
    if "trap" in p:
        return "trap"
    if "house" in p:
        return "house"
    if "lofi" in p or "lo-fi" in p:
        return "lofi"
    return "electronic"


def build_sections(total_bars: int) -> list[tuple[str, int, int, float]]:
    # intro, verse, hook/drop, break, drop, outro
    parts: list[tuple[str, float]] = [
        ("intro", 0.12),
        ("verse", 0.23),
        ("hook", 0.20),
        ("break", 0.10),
        ("drop", 0.25),
        ("outro", 0.10),
    ]
    energies = {
        "intro": 0.45,
        "verse": 0.68,
        "hook": 0.9,
        "break": 0.35,
        "drop": 1.0,
        "outro": 0.5,
    }
    bars_acc = 0
    out: list[tuple[str, int, int, float]] = []
    for idx, (name, ratio) in enumerate(parts):
        if idx == len(parts) - 1:
            span = total_bars - bars_acc
        else:
            span = max(4, int(total_bars * ratio))
        start = bars_acc
        end = min(total_bars, start + span)
        out.append((name, start, end, energies[name]))
        bars_acc = end
    if out[-1][2] < total_bars:
        name, s, _, e = out[-1]
        out[-1] = (name, s, total_bars, e)
    return out


def section_energy(bar_idx: int, sections: list[tuple[str, int, int, float]]) -> float:
    for _, s, e, en in sections:
        if s <= bar_idx < e:
            return en
    return 0.7


def uk_drill_snare_steps() -> set[int]:
    # Half-time clap/snare feel
    return {8, 12}


def choose_kick_pattern(style: str, rng: random.Random) -> list[int]:
    if style in ("uk_drill", "drill"):
        patterns = [
            [0, 5, 10, 14],
            [0, 3, 7, 10, 13],
            [0, 6, 9, 12, 15],
            [0, 4, 8, 11, 14],
        ]
        return patterns[rng.randrange(len(patterns))]
    if style == "trap":
        return [0, 6, 8, 11, 14]
    if style == "house":
        return [0, 4, 8, 12]
    return [0, 4, 9, 12]


def get_cloud_song_plan(backend: str, prompt: str, seconds: int) -> dict:
    if backend != "gemini":
        return {}
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        return {}
    ask = (
        "Return strict JSON only with keys: style,bpm,mood,key_root,energy."
        " style one of [uk_drill,drill,trap,house,lofi,electronic]."
        " bpm integer 60-180. energy number 0.2-1.0."
        f" Prompt: {prompt}. Duration seconds: {seconds}."
    )
    body = {"contents": [{"parts": [{"text": ask}]}]}
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        parsed = json.loads(resp.read().decode("utf-8", "ignore"))
    text = (
        parsed.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "{}")
    )
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


def load_profile(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def profile_for_prompt(prompt: str, profile: dict[str, Any]) -> dict[str, Any]:
    if not profile:
        return {}
    out: dict[str, Any] = {}
    if "bpm" not in prompt.lower():
        pbpm = safe_int(profile.get("preferred_bpm", 0), 0)
        if pbpm > 0:
            out["bpm"] = pbpm
    out["drive"] = clamp(safe_float(profile.get("preferred_drive", 1.2), 1.2), 0.6, 2.5)
    out["guidance"] = clamp(safe_float(profile.get("preferred_guidance", 3.5), 3.5), 1.0, 6.0)
    out["temperature"] = clamp(safe_float(profile.get("preferred_temperature", 1.0), 1.0), 0.2, 2.0)
    if infer_style(prompt) == "electronic" and profile.get("default_style"):
        out["style"] = str(profile.get("default_style")).strip().lower()
    return out


def write_wav_from_float(path: Path, sample_rate: int, channels: int, interleaved: list[float]) -> None:
    ints_out = [max(-32768, min(32767, int(clamp(x, -1.0, 1.0) * 32767.0))) for x in interleaved]
    raw_out = struct.pack("<" + "h" * len(ints_out), *ints_out)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(raw_out)


def can_use_musicgen() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
        from transformers import AutoProcessor, MusicgenForConditionalGeneration  # noqa: F401
    except Exception as exc:
        return False, str(exc)
    return True, "ok"


def render_musicgen_song(
    prompt: str,
    seconds: int,
    sample_rate: int,
    channels: int,
    model_id: str,
    guidance: float,
    temperature: float,
    chunk_seconds: int,
) -> tuple[list[float], dict[str, Any]]:
    import numpy as np
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device)

    total_samples = seconds * sample_rate
    max_chunk = max(6, min(30, int(chunk_seconds)))
    chunks: list[np.ndarray] = []
    tokens_per_second = 50
    remaining = seconds
    idx = 0
    while remaining > 0:
        idx += 1
        sec = min(max_chunk, remaining)
        remaining -= sec
        section_prompt = (
            f"{prompt}. section {idx}. coherent arrangement, strong groove, polished mix."
        )
        inputs = processor(text=[section_prompt], padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        audio_values = model.generate(
            **inputs,
            do_sample=True,
            guidance_scale=float(guidance),
            temperature=float(temperature),
            max_new_tokens=int(sec * tokens_per_second),
        )
        arr = audio_values[0].detach().float().cpu().numpy()
        if arr.ndim == 2:
            arr = arr[0]
        chunks.append(arr)

    generated_sr = int(getattr(model.config.audio_encoder, "sampling_rate", 32000))
    if not chunks:
        return [0.0] * (total_samples * channels), {"generated_sample_rate": generated_sr}
    merged = chunks[0]
    for nxt in chunks[1:]:
        merged = np.concatenate([merged, nxt])
    if generated_sr != sample_rate:
        x_old = np.linspace(0.0, 1.0, num=len(merged), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=int(len(merged) * sample_rate / generated_sr), endpoint=False)
        merged = np.interp(x_new, x_old, merged)
    if len(merged) < total_samples:
        merged = np.pad(merged, (0, total_samples - len(merged)))
    else:
        merged = merged[:total_samples]
    if channels == 1:
        interleaved = merged.tolist()
    else:
        out = np.empty(total_samples * 2, dtype=np.float32)
        out[0::2] = merged
        out[1::2] = np.roll(merged, 64) * 0.98
        interleaved = out.tolist()
    return interleaved, {
        "generated_sample_rate": generated_sr,
        "model": model_id,
        "device": device,
    }


def run_generate_mode(args: argparse.Namespace) -> int:
    prompt = args.prompt.strip() or "energetic electronic track"
    feedback_path, labeled_path, profile_path, generations_path = default_data_paths(args)
    if args.auto_learn and not profile_path.exists():
        seed_profile = build_learning_profile(read_jsonl(feedback_path), read_jsonl(labeled_path))
        if seed_profile.get("examples", 0) > 0:
            write_json(profile_path, seed_profile)
    profile = load_profile(profile_path)
    learned = profile_for_prompt(prompt, profile)

    seed = pick_seed(prompt)
    rng = random.Random(seed)

    sr = int(args.sample_rate)
    channels = max(1, min(2, int(args.channels)))
    seconds = max(4, min(300, int(args.generate_seconds)))
    style = str(learned.get("style", infer_style(prompt))).strip().lower()

    cloud_plan = {}
    if args.backend == "gemini":
        try:
            cloud_plan = get_cloud_song_plan(args.backend, prompt, seconds)
        except Exception as exc:
            print(f"warning: cloud plan failed ({exc}); using local planner", file=sys.stderr)

    if "style" in cloud_plan:
        style = str(cloud_plan["style"]).strip().lower() or style

    if args.generate_bpm:
        bpm = int(args.generate_bpm)
    elif "bpm" in cloud_plan:
        try:
            bpm = int(cloud_plan["bpm"])
        except Exception:
            bpm = parse_bpm(prompt, rng.randint(110, 150))
    elif "bpm" in learned:
        bpm = safe_int(learned["bpm"], parse_bpm(prompt, rng.randint(110, 150)))
    else:
        default_bpm = 142 if style in ("uk_drill", "drill") else 140 if style == "trap" else 124 if style == "house" else 90 if style == "lofi" else 128
        bpm = parse_bpm(prompt, default_bpm)

    bpm = max(60, min(180, bpm))
    guidance = clamp(safe_float(getattr(args, "musicgen_guidance", 3.5), safe_float(learned.get("guidance", 3.5), 3.5)), 1.0, 6.0)
    temperature = clamp(safe_float(getattr(args, "musicgen_temperature", 1.0), safe_float(learned.get("temperature", 1.0), 1.0)), 0.2, 2.0)
    engine = args.engine
    if engine == "auto":
        engine = "musicgen" if args.backend == "musicgen" else "synth"

    if engine == "musicgen":
        ok, reason = can_use_musicgen()
        if not ok:
            print(f"warning: musicgen unavailable ({reason}); falling back to synth", file=sys.stderr)
            engine = "synth"
        else:
            try:
                interleaved, meta = render_musicgen_song(
                    prompt=prompt,
                    seconds=seconds,
                    sample_rate=sr,
                    channels=channels,
                    model_id=args.musicgen_model,
                    guidance=guidance,
                    temperature=temperature,
                    chunk_seconds=args.musicgen_chunk_seconds,
                )
                write_wav_from_float(Path(args.generate_output), sr, channels, interleaved)
                out_obj = {
                    "generate_output": args.generate_output,
                    "prompt": prompt,
                    "backend": args.backend,
                    "engine": "musicgen",
                    "style": style,
                    "seconds": seconds,
                    "bpm": bpm,
                    "sample_rate": sr,
                    "channels": channels,
                    "guidance": guidance,
                    "temperature": temperature,
                }
                out_obj.update(meta)
                append_jsonl(generations_path, {"timestamp": now_iso(), **out_obj})
                if args.auto_learn:
                    refreshed = build_learning_profile(read_jsonl(feedback_path), read_jsonl(labeled_path))
                    if refreshed.get("examples", 0) > 0:
                        write_json(profile_path, refreshed)
                print(json.dumps(out_obj))
                return 0
            except Exception as exc:
                print(f"warning: musicgen generation failed ({exc}); falling back to synth", file=sys.stderr)

    beat_s = 60.0 / bpm
    bar_s = beat_s * 4.0
    n = int(seconds * sr)
    total_bars = max(4, int(seconds / bar_s))
    sections = build_sections(total_bars)
    master_energy = 0.8
    if "energy" in cloud_plan:
        try:
            master_energy = clamp(float(cloud_plan["energy"]), 0.2, 1.0)
        except Exception:
            pass

    left = [0.0] * n
    right = [0.0] * n

    root_choices = [45, 47, 48, 50, 52]
    base_root = rng.choice(root_choices)
    progression = rng.choice([[0, 3, 5, 2], [0, 5, 3, 4], [0, 2, 5, 3], [0, 7, 5, 3]])
    snare_steps = uk_drill_snare_steps() if style in ("uk_drill", "drill") else {4, 12}

    # Render per bar with section-aware density.
    for bar in range(total_bars):
        sec_energy = section_energy(bar, sections) * master_energy
        kick_pattern = choose_kick_pattern(style, rng if (bar % 4 == 0) else random.Random(seed + bar))
        bar_start_t = bar * bar_s
        prog = progression[bar % len(progression)]
        root = base_root + prog
        bass_note = root - 12

        # Drums on 16th grid
        for step in range(16):
            t0 = bar_start_t + step * (bar_s / 16.0)
            i0 = int(t0 * sr)
            if i0 >= n:
                continue

            is_kick = step in kick_pattern and rng.random() < (0.7 + 0.3 * sec_energy)
            is_snare = step in snare_steps and rng.random() < (0.85 + 0.15 * sec_energy)
            is_hat = (step % 2 == 0) if style in ("house", "lofi") else (step % 1 == 0)

            if is_kick:
                kl = int(0.20 * sr)
                for j in range(kl):
                    i = i0 + j
                    if i >= n:
                        break
                    t = j / sr
                    env = math.exp(-t * 22.0)
                    freq = 150.0 - 105.0 * min(1.0, t * 9.0)
                    s = math.sin(2.0 * math.pi * freq * t) * env * (0.78 + 0.22 * sec_energy)
                    left[i] += s
                    right[i] += s

            if is_snare:
                sl = int(0.15 * sr)
                for j in range(sl):
                    i = i0 + j
                    if i >= n:
                        break
                    t = j / sr
                    env = math.exp(-t * 35.0)
                    noise = (rng.random() * 2.0 - 1.0) * env * 0.30
                    tone = math.sin(2.0 * math.pi * 190.0 * t) * env * 0.14
                    s = (noise + tone) * (0.8 + 0.2 * sec_energy)
                    left[i] += s
                    right[i] += s

            if is_hat and rng.random() < (0.75 + 0.2 * sec_energy):
                hl = int(0.03 * sr)
                pan = -0.25 if (step % 2 == 0) else 0.25
                for j in range(hl):
                    i = i0 + j
                    if i >= n:
                        break
                    t = j / sr
                    env = math.exp(-t * 95.0)
                    s = (rng.random() * 2.0 - 1.0) * env * (0.08 + 0.05 * sec_energy)
                    left[i] += s * (1.0 - pan)
                    right[i] += s * (1.0 + pan)

                # Drill/trap micro-roll
                if style in ("uk_drill", "drill", "trap") and step in (7, 15) and rng.random() < 0.35:
                    roll_step = int((bar_s / 48.0) * sr)
                    for rr in range(2):
                        r0 = i0 + rr * roll_step
                        for j in range(int(0.018 * sr)):
                            i = r0 + j
                            if i >= n:
                                break
                            t = j / sr
                            env = math.exp(-t * 120.0)
                            s = (rng.random() * 2.0 - 1.0) * env * 0.06
                            left[i] += s
                            right[i] += s

        # Bass/808 with mild glide for drill/trap.
        for step in range(8):
            t0 = bar_start_t + step * (bar_s / 8.0)
            i0 = int(t0 * sr)
            if i0 >= n:
                continue
            seg = int((bar_s / 8.0) * sr)
            gate = 0.6 + 0.4 * (1 if step % 2 == 0 else 0.7)
            note_a = bass_note + rng.choice([0, 0, 3, 5, -2])
            note_b = note_a + rng.choice([0, 0, 2, -2]) if style in ("uk_drill", "drill", "trap") and rng.random() < 0.35 else note_a
            f_a = midi_to_hz(note_a)
            f_b = midi_to_hz(note_b)
            for j in range(seg):
                i = i0 + j
                if i >= n:
                    break
                p = j / max(1, seg - 1)
                # portamento-style slide
                f = f_a + (f_b - f_a) * (p * p)
                tt = (t0 + j / sr)
                bass = (math.sin(2.0 * math.pi * f * tt) * 0.65 + math.sin(2.0 * math.pi * (f * 0.5) * tt) * 0.35)
                amp = 0.16 * gate * (0.6 + 0.4 * sec_energy)
                s = bass * amp
                left[i] += s
                right[i] += s

        # Sparse top melody/pad, more active in hook/drop.
        if sec_energy > 0.6:
            lead_note = root + rng.choice([12, 14, 15, 17, 19])
            f = midi_to_hz(lead_note)
            for j in range(int(bar_s * sr)):
                i = int(bar_start_t * sr) + j
                if i >= n:
                    break
                tt = j / sr
                lfo = 0.5 + 0.5 * math.sin(2 * math.pi * 0.25 * tt)
                s = math.sin(2.0 * math.pi * f * (bar_start_t + tt)) * 0.05 * lfo * sec_energy
                left[i] += s * 0.8
                right[i] += s * 1.2

    # Global sidechain pump
    for beat in range(int(seconds / beat_s) + 1):
        start = int(beat * beat_s * sr)
        pump_len = int(0.20 * sr)
        for j in range(pump_len):
            i = start + j
            if i >= n:
                break
            amt = (0.2 if style == "lofi" else 0.3) * math.exp(-j / (0.08 * sr))
            left[i] *= (1.0 - amt)
            right[i] *= (1.0 - amt)

    # Normalize + saturation drive
    peak = 1e-9
    for i in range(n):
        peak = max(peak, abs(left[i]), abs(right[i]))
    norm = 0.92 / peak

    drive = safe_float(learned.get("drive", 1.2), 1.2)
    if args.backend in ("gemini", "deepseek"):
        try:
            drive = get_cloud_drive_hint(args.backend, prompt)
        except Exception as exc:
            print(f"warning: cloud hint failed ({exc}); using local generation drive", file=sys.stderr)

    interleaved: list[float] = []
    for i in range(n):
        l = math.tanh(clamp(left[i] * norm, -1.0, 1.0) * drive)
        r = math.tanh(clamp(right[i] * norm, -1.0, 1.0) * drive)
        if channels == 1:
            interleaved.append((l + r) * 0.5)
        else:
            interleaved.extend([l, r])

    write_wav_from_float(Path(args.generate_output), sr, channels, interleaved)
    out_obj = {
        "generate_output": args.generate_output,
        "prompt": prompt,
        "backend": args.backend,
        "engine": "synth",
        "style": style,
        "seconds": seconds,
        "bpm": bpm,
        "sample_rate": sr,
        "channels": channels,
        "sections": [{"name": n, "start_bar": s, "end_bar": e, "energy": en} for n, s, e, en in sections],
        "applied_drive": drive,
        "guidance": guidance,
        "temperature": temperature,
    }
    append_jsonl(generations_path, {"timestamp": now_iso(), **out_obj})
    if args.auto_learn:
        refreshed = build_learning_profile(read_jsonl(feedback_path), read_jsonl(labeled_path))
        if refreshed.get("examples", 0) > 0:
            write_json(profile_path, refreshed)
    print(json.dumps(out_obj))
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


def run_feedback_mode(args: argparse.Namespace) -> int:
    feedback_path, labeled_path, profile_path, _ = default_data_paths(args)
    if not args.feedback_track:
        print("feedback mode requires --feedback-track", file=sys.stderr)
        return 11
    item = {
        "timestamp": now_iso(),
        "track": args.feedback_track,
        "prompt": args.prompt,
        "rating": clamp(float(args.feedback_rating), 1.0, 5.0),
        "notes": args.feedback_notes,
        "style": args.feedback_style or infer_style(args.prompt),
        "bpm": args.feedback_bpm,
        "drive": args.feedback_drive,
        "guidance": args.feedback_guidance,
        "temperature": args.feedback_temperature,
    }
    append_jsonl(feedback_path, item)
    if args.auto_learn:
        prof = build_learning_profile(read_jsonl(feedback_path), read_jsonl(labeled_path))
        write_json(profile_path, prof)
    print(json.dumps({"feedback_logged": True, "file": str(feedback_path)}))
    return 0


def run_export_profile_mode(args: argparse.Namespace) -> int:
    feedback_path, labeled_path, profile_path, _ = default_data_paths(args)
    prof = build_learning_profile(read_jsonl(feedback_path), read_jsonl(labeled_path))
    out = Path(args.learn_export) if args.learn_export else profile_path
    write_json(out, prof)
    print(json.dumps({"profile_exported": str(out), "examples": prof.get("examples", 0)}))
    return 0


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
    parser.add_argument("--backend", choices=["local", "gemini", "deepseek", "musicgen"], default="local")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--offline-input")
    parser.add_argument("--offline-output")
    parser.add_argument("--generate-output")
    parser.add_argument("--generate-seconds", type=int, default=24)
    parser.add_argument("--generate-bpm", type=int)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--drive", type=float, default=1.2)
    parser.add_argument("--engine", choices=["auto", "synth", "musicgen"], default="auto")
    parser.add_argument("--musicgen-model", default="facebook/musicgen-small")
    parser.add_argument("--musicgen-guidance", type=float, default=3.5)
    parser.add_argument("--musicgen-temperature", type=float, default=1.0)
    parser.add_argument("--musicgen-chunk-seconds", type=int, default=18)

    parser.add_argument("--data-dir", default="")
    parser.add_argument("--feedback-path", default="")
    parser.add_argument("--labeled-path", default="")
    parser.add_argument("--profile-path", default="")
    parser.add_argument("--auto-learn", action="store_true")

    parser.add_argument("--feedback-track", default="")
    parser.add_argument("--feedback-rating", type=float, default=3.0)
    parser.add_argument("--feedback-notes", default="")
    parser.add_argument("--feedback-style", default="")
    parser.add_argument("--feedback-bpm", type=int, default=0)
    parser.add_argument("--feedback-drive", type=float, default=0.0)
    parser.add_argument("--feedback-guidance", type=float, default=0.0)
    parser.add_argument("--feedback-temperature", type=float, default=0.0)
    parser.add_argument("--learn-export", default="")
    args = parser.parse_args()

    if args.learn_export:
        return run_export_profile_mode(args)
    if args.feedback_track:
        return run_feedback_mode(args)

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
