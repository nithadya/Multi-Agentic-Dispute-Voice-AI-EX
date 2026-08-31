"""
Per-participant session tracking + barge-in helpers.

The voice worker can host many concurrent rooms (each with its own
participant). ``SessionManager`` keeps a small in-memory record per
participant — turn count, last-activity timestamp, who's currently
speaking — so barge-in events have the context they need.

Nothing here talks to LiveKit directly; these are plain dataclasses
that the agent factory updates in response to LiveKit events.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from loguru import logger


# ── Session state ──────────────────────────────────────────────

@dataclass
class VoiceSession:
    """Per-participant voice session record."""

    participant_id: str
    user_id: str
    session_id: str
    connected_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    turn_count: int = 0
    is_agent_speaking: bool = False
    is_user_speaking: bool = False

    def record_turn(self) -> None:
        self.turn_count += 1
        self.last_activity = time.time()

    @property
    def duration_seconds(self) -> float:
        return time.time() - self.connected_at


# ── Manager ────────────────────────────────────────────────────

class SessionManager:
    """Maps LiveKit participant identities to voice sessions.

    Used by the agent factory so callbacks (``on_user_started_speaking``
    etc.) have a place to record state per call.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, VoiceSession] = {}

    def get_or_create(
        self,
        *,
        participant_id: str,
        user_id: str,
        room_name: str,
    ) -> VoiceSession:
        if participant_id in self._sessions:
            return self._sessions[participant_id]

        session = VoiceSession(
            participant_id=participant_id,
            user_id=user_id,
            session_id=f"voice-{room_name}",
        )
        self._sessions[participant_id] = session
        logger.info(f"Voice session created — user={user_id} session={session.session_id}")
        return session

    def end_session(self, participant_id: str) -> Optional[VoiceSession]:
        session = self._sessions.pop(participant_id, None)
        if session:
            logger.info(
                f"Voice session ended — user={session.user_id} "
                f"turns={session.turn_count} duration={session.duration_seconds:.1f}s"
            )
        return session

    @property
    def active_count(self) -> int:
        return len(self._sessions)


# ── Event helpers ──────────────────────────────────────────────
# These are tiny pure functions that mutate session state. They exist
# so the agent factory's event handlers can stay one-liners.

def on_user_speech_started(session: VoiceSession) -> None:
    session.is_user_speaking = True
    session.last_activity = time.time()
    if session.is_agent_speaking:
        logger.debug("Barge-in detected — user spoke during agent playback")


def on_user_speech_committed(session: VoiceSession, transcript: str) -> None:
    session.is_user_speaking = False
    session.record_turn()
    preview = transcript[:80] + ("..." if len(transcript) > 80 else "")
    logger.info(f'[turn {session.turn_count}] user: "{preview}"')


def on_agent_speech_started(session: VoiceSession) -> None:
    session.is_agent_speaking = True


def on_agent_speech_interrupted(session: VoiceSession) -> None:
    """User barged in. LiveKit has already stopped TTS — we just record it."""
    session.is_agent_speaking = False
    logger.info(f"Agent interrupted — session={session.session_id} turn={session.turn_count}")


def on_agent_speech_finished(session: VoiceSession) -> None:
    session.is_agent_speaking = False
