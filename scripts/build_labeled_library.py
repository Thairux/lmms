#!/usr/bin/env python3
"""Build a labeled JSONL library from filenames in an audio folder."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def infer_style(name: str) -> str:
    n = name.lower()
    if "ukdrill" in n or "uk_drill" in n or ("uk" in n and "drill" in n):
        return "uk_drill"
    if "drill" in n:
        return "drill"
    if "trap" in n:
        return "trap"
    if "house" in n:
        return "house"
    if "lofi" in n or "lo-fi" in n:
        return "lofi"
    if "boombap" in n or "boom_bap" in n:
        return "boom_bap"
    if "edm" in n:
        return "edm"
    return "electronic"


def infer_bpm(name: str) -> int:
    m = re.search(r"(\d{2,3})\s*bpm", name.lower())
    if m:
        return max(60, min(200, int(m.group(1))))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", default="data/ai_training/labeled_prompts.jsonl")
    parser.add_argument("--default-quality", type=int, default=3)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sorted(in_dir.glob("*")):
        if p.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg"}:
            continue
        name = p.stem.replace("_", " ")
        row = {
            "prompt": name,
            "style": infer_style(name),
            "bpm": infer_bpm(name),
            "quality": max(1, min(5, int(args.default_quality))),
            "source_file": str(p),
        }
        rows.append(row)

    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps({"written": len(rows), "output": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
