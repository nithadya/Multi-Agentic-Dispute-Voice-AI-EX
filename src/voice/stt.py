"""
STT adapter — Deepgram Nova-3 via the LiveKit plugin.

Kept as a separate file (rather than inlined in ``agent.py``) so students
can read it as a self-contained "STT factory" example. Switching STT
providers (e.g. to Whisper or AssemblyAI) is a single-file change.
"""

from __future__ import annotations

from livekit.plugins import deepgram

from voice.config import VoiceConfig


def make_stt(cfg: VoiceConfig):
    """Build the Deepgram streaming STT plugin from ``cfg``.

    The LiveKit Deepgram plugin handles the websocket lifecycle, audio
    framing, and interim/final result events. We just hand it the model
    name, language, and let it stream.
    """
    return deepgram.STT(
        model=cfg.stt_model,
        language=cfg.stt_language,
    )
