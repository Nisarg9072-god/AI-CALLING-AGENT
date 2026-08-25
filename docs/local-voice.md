# Local Voice Mode

Local Voice mode lets you test the full voice AI without paying for telephony providers, cloud STT, or cloud TTS. Everything runs on your local machine.

## Prerequisites

1. Install system dependencies:
   - Linux: `sudo apt install portaudio19-dev`
   - macOS: `brew install portaudio`

2. Install Python dependencies:
   ```bash
   uv pip install -e ".[voice]"
   ```
   *(This installs `faster-whisper`, `piper-tts`, and `sounddevice`)*

3. Download a Piper voice model:
   ```bash
   mkdir -p data/voices
   # Download the model and JSON config
   curl -L -o data/voices/en_US-amy-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
   curl -L -o data/voices/en_US-amy-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json
   ```

## Running the Web UI

1. Start the API server:
   ```bash
   uv run uvicorn app.api.main:app --reload
   ```
2. Open http://localhost:8000/ui in your browser.
3. Click **Start Call** and hold the **Record** button to speak.
4. Watch the agent's thought process (Observe→Decide→Act) in real-time in the Agent Trace panel.

## Running the CLI Voice Test

To run a pure terminal-based voice test (no browser):
```bash
uv run python -m cli.voice_test
```

## Configuration

In your `.env` file:
```env
VOICE_TRANSPORT=local
WHISPER_MODEL=small
WHISPER_DEVICE=auto
PIPER_MODEL_PATH=./data/voices/en_US-amy-medium.onnx
```
