"""
Voice worker entrypoint.

Run this to start a LiveKit agent worker that joins rooms and runs the
full Silero VAD → Deepgram STT → LangGraph → Deepgram TTS pipeline.

Usage::

    # native (foreground)
    PYTHONPATH=src python -m voice.run

    # via Makefile
    make voice

    # via Docker
    make demo-voice

Required env vars (loaded from .env):
    LIVEKIT_URL          (e.g. wss://your-project.livekit.cloud)
    LIVEKIT_API_KEY
    LIVEKIT_API_SECRET
    DEEPGRAM_API_KEY
"""

from __future__ import annotations

import os
import sys

# Ensure src/ is importable when running ``python src/voice/run.py``
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from dotenv import find_dotenv, load_dotenv

# Walk up from cwd to find .env so `make voice` / direct python works
# regardless of where it's invoked from. Without this, dotenv silently
# does nothing if cwd isn't the project root, and Supabase init at
# import-time fails with "SUPABASE_DB_URL must be set" noise in the logs.
load_dotenv(find_dotenv(usecwd=True))

from livekit.agents import JobContext, WorkerOptions, cli
from loguru import logger

from infrastructure.log import setup_logging
from voice.agent import create_and_start_agent
from voice.config import load_voice_config, validate_voice_env


async def entrypoint(ctx: JobContext) -> None:
    """Called by LiveKit for each new room session."""
    await create_and_start_agent(ctx)


def main() -> None:
    setup_logging()
    cfg = load_voice_config()

    print()
    print("  DisputeAI — Voice Worker")
    print(f"  {'-' * 48}")
    print(f"  LiveKit URL  : {cfg.livekit_url or '(unset)'}")
    print(f"  STT          : {cfg.stt_provider}/{cfg.stt_model}")
    print(f"  TTS          : {cfg.tts_provider}/{cfg.tts_model}")
    print(f"  VAD threshold: {cfg.vad_threshold}")
    print(f"  Silence      : {cfg.silence_threshold_ms} ms")
    print(f"  Endpointing  : {cfg.min_endpointing_delay} s")
    print(f"  Interruptions: {cfg.interruption_enabled}")
    print(f"  {'-' * 48}")
    print()

    validate_voice_env()
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            initialize_process_timeout=120.0,
            num_idle_processes=1,
        )
    )


if __name__ == "__main__":
    main()
