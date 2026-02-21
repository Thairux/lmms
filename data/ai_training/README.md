# AI Training Library

This folder stores local data used by the AI bridge to improve generations over time.

## Files

- `labeled_prompts.jsonl`: starter labeled prompt library (style/BPM/quality targets).
- `feedback.jsonl`: your ratings and notes from generated tracks.
- `profile.json`: learned profile computed from labeled prompts + feedback.
- `generations.jsonl`: generation history metadata.

## JSONL row format

Each line is one JSON object, for example:

```json
{"prompt":"UK drill beat, dark 808, 142 bpm","style":"uk_drill","bpm":142,"quality":5}
```
