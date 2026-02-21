# AIStudio Tool Plugin

`AIStudio` is a Tool plugin that generates arrangement and mix ideas from text prompts.

## Location

- Plugin sources: `plugins/AIStudio/`
- Main files:
  - `plugins/AIStudio/AIStudio.cpp`
  - `plugins/AIStudio/AIStudio.h`
  - `plugins/AIStudio/AIStudioView.cpp`
  - `plugins/AIStudio/AIStudioView.h`

## Providers

`AIStudio` supports:

- `Local` (free, no key)
- `Gemini API`
- `DeepSeek API`

## Keys

You can enter a key in the plugin UI, or use environment variables:

- `GEMINI_API_KEY`
- `DEEPSEEK_API_KEY`

## Notes

- Local mode is always available and does not require downloading any model.
- API-backed modes require network access and provider credentials.
