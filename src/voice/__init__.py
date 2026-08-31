"""
Voice pipeline (Week 14) — LiveKit + Deepgram + Silero VAD.

A self-contained side-car module that calls into the existing
``AgentOrchestrator`` via ``orchestrator.achat()``. Adding voice does
NOT modify ``api/routers/chat.py`` or ``agents/orchestrator.py``.

Pipeline:

    Mic ──▶ Silero VAD ──▶ Deepgram STT ──▶ LangGraphLLMAdapter ──▶ Deepgram TTS ──▶ Speaker
                                                  │
                                                  ▼
                                  AgentOrchestrator.achat()  (existing multi-agent graph)

Public API
----------
VoiceConfig             Configuration dataclass (STT/TTS/VAD/EOU)
load_voice_config       Read voice settings from param.yaml + env
validate_voice_env      Check required env vars are present
LangGraphLLMAdapter     Bridges AgentOrchestrator ↔ LiveKit LLM interface
build_voice_agent       Build a LiveKit Agent for one participant
create_and_start_agent  Worker entrypoint helper
SessionManager          Per-participant voice session tracking
"""

from voice.adapter import LangGraphLLMAdapter
from voice.agent import build_voice_agent, create_and_start_agent
from voice.config import VoiceConfig, load_voice_config, validate_voice_env
from voice.pipeline import SessionManager, VoiceSession

__all__ = [
    "VoiceConfig",
    "load_voice_config",
    "validate_voice_env",
    "LangGraphLLMAdapter",
    "build_voice_agent",
    "create_and_start_agent",
    "SessionManager",
    "VoiceSession",
]
