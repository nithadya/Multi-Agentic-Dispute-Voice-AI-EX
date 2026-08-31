"""
TTS adapter — provider-dispatched factory.

Two providers supported, switchable via ``param.yaml`` ``voice.tts_provider``:

  - ``elevenlabs`` (default) — premium voices, ~250 ms TTFB, requires ELEVEN_API_KEY
  - ``deepgram``              — Aura 2, ~200 ms TTFB, uses DEEPGRAM_API_KEY

Kept as a separate file (rather than inlined in ``agent.py``) so students
can read it as a self-contained TTS factory and so swapping providers is
a single-file change.
"""

from __future__ import annotations

from loguru import logger

from voice.config import VoiceConfig


def make_tts(cfg: VoiceConfig):
    """Build the streaming TTS plugin for the configured provider.

    Both plugins return audio chunks the moment synthesis starts, so the
    user hears the beginning of the response while the rest is still
    being generated.
    """
    provider = cfg.tts_provider.lower()

    if provider == "elevenlabs":
        # Lazy import — keeps the module importable even when the
        # elevenlabs plugin isn't installed (e.g. when teaching the
        # standalone notebook 03 with only the Deepgram SDK).
        from livekit.plugins import elevenlabs

        if not cfg.eleven_api_key:
            raise EnvironmentError(
                "ELEVEN_API_KEY not set — required for tts_provider=elevenlabs. "
                "Get one from https://elevenlabs.io/app/settings/api-keys "
                "and add it to your .env file."
            )

        logger.debug(
            f"TTS = ElevenLabs (model={cfg.tts_model}, voice_id={cfg.tts_voice_id})"
        )
        return elevenlabs.TTS(
            model=cfg.tts_model,
            voice_id=cfg.tts_voice_id,
            api_key=cfg.eleven_api_key,
        )

    if provider == "deepgram":
        from livekit.plugins import deepgram

        logger.debug(f"TTS = Deepgram (model={cfg.tts_model})")
        return deepgram.TTS(model=cfg.tts_model)

    raise ValueError(
        f"Unknown tts_provider: {cfg.tts_provider!r}. "
        f"Supported: 'elevenlabs', 'deepgram'."
    )
