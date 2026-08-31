"""
LangGraph ↔ LiveKit adapter.

This file is the only place the voice layer touches the agent. Everything
else (STT, TTS, VAD, session) is generic LiveKit plumbing — this adapter
is what makes it *our* agent answering the call.

Architecture:

    LiveKit Agent  ──── audio ───▶ Deepgram STT
                                     │ transcript
                                     ▼
                              LangGraphLLMAdapter   ◀── this file
                                     │
                                     ▼
                  AgentOrchestrator.achat(text, user_id, session_id)
                                     │ AgentResponse.answer
                                     ▼
                               Deepgram TTS  ──── audio ───▶ user

Future improvement (Week 14+):
    Right now we call ``orchestrator.achat()``, which uses the legacy
    multi-agent graph (recall → supervisor → agents → merge → save).
    The HTTP chat endpoint (``api/routers/chat.py``) goes through the
    newer ``decision_graph`` first (parallel guardrail + router + CAG)
    before falling through to the same multi-agent graph. Voice users
    therefore skip the guardrail and CAG cache. To lift this, extract
    ``_run_chat_pipeline`` from chat.py into a shared service module
    and call it here. Kept simple for now to keep the voice module
    fully decoupled.

Compatible with **livekit-agents >= 1.5.0**.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from loguru import logger

from livekit.agents import (
    APIConnectOptions,
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.llm import (
    ChatChunk,
    ChatContext,
    ChoiceDelta,
    LLM,
    LLMStream,
    Tool,
)

from agents.orchestrator import AgentOrchestrator


# ── Adapter ────────────────────────────────────────────────────

class LangGraphLLMAdapter(LLM):
    """Wrap ``AgentOrchestrator`` so it satisfies LiveKit's ``LLM`` interface.

    LiveKit's ``Agent`` calls ``llm.chat(chat_ctx=...)`` after STT
    finalises a transcript. This adapter hands that transcript to the
    orchestrator and returns the answer text as a single ``ChatChunk``
    for the TTS plugin to speak.

    Parameters
    ----------
    orchestrator : AgentOrchestrator
        The pre-built multi-agent graph (from ``build_agent()``).
    user_id : str
        Caller identity. Defaults to ``"voice-user"`` when no
        participant identity is available; the agent factory
        overrides this with ``participant.identity`` per session.
    session_id : str
        Voice session identifier. Defaults to ``"voice-session"``;
        the agent factory uses ``f"voice-{room.name}"``.
    """

    def __init__(
        self,
        orchestrator: AgentOrchestrator,
        user_id: str = "voice-user",
        session_id: str = "voice-session",
    ) -> None:
        super().__init__()
        self._orchestrator = orchestrator
        self._user_id = user_id
        self._session_id = session_id
        self._current_task: Optional[asyncio.Task] = None

    # ── LiveKit LLM interface (v1.5) ───────────────────────────

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[Any] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> "LangGraphLLMStream":
        """Called by LiveKit after STT produces a final transcript."""
        return LangGraphLLMStream(
            llm=self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            orchestrator=self._orchestrator,
            user_id=self._user_id,
            session_id=self._session_id,
        )

    # ── Cancellation hook (barge-in) ───────────────────────────

    def cancel_current(self) -> None:
        """Cancel any in-flight ``achat()`` task. Called on barge-in."""
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            logger.info("Cancelled in-flight agent task (barge-in)")

    def update_identity(self, user_id: str, session_id: str) -> None:
        """Update caller identity (used when a new participant joins)."""
        self._user_id = user_id
        self._session_id = session_id
        logger.debug(f"Adapter identity → user={user_id} session={session_id}")


# ── Stream ─────────────────────────────────────────────────────

class LangGraphLLMStream(LLMStream):
    """Async iterator that runs the agent and yields its answer.

    LiveKit's pipeline consumes this stream chunk-by-chunk and pipes
    each chunk's content into the TTS plugin. Since ``achat()`` returns
    a complete answer (not a token stream), we emit a single chunk
    containing the full response. The TTS plugin handles word-level
    streaming downstream.
    """

    def __init__(
        self,
        llm: LangGraphLLMAdapter,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool],
        conn_options: APIConnectOptions,
        orchestrator: AgentOrchestrator,
        user_id: str,
        session_id: str,
    ) -> None:
        super().__init__(
            llm=llm,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
        )
        self._orchestrator = orchestrator
        self._user_id = user_id
        self._session_id = session_id
        self._adapter = llm

    async def _run(self) -> None:
        # ── Extract the latest user message from the chat context ──
        # livekit-agents 1.5.x: ``chat_ctx.items`` is the canonical list of
        # ChatItem objects (messages + tool calls). ``chat_ctx.messages`` is
        # now a *method* (returns a filtered iterator) and is not reversible.
        # Always reach for ``items`` and filter to role=user ourselves.
        items = getattr(self._chat_ctx, "items", None)
        if items is None:
            # Defensive fallback for any earlier SDK that still has the
            # ``messages`` attribute as a list. Newer SDKs hit the path above.
            messages = getattr(self._chat_ctx, "messages", None)
            items = messages() if callable(messages) else (messages or [])

        user_text = ""
        for msg in reversed(list(items)):
            if getattr(msg, "role", None) != "user":
                continue
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                user_text = content
            elif isinstance(content, list):
                # In 1.5.x, ``content`` is always a list. Stitch the string
                # parts together so phrases broken across parts survive.
                user_text = " ".join(p for p in content if isinstance(p, str)).strip()
            if user_text:
                break

        if not user_text:
            logger.warning("No user text found in chat context — skipping agent")
            return

        preview = user_text[:80] + ("..." if len(user_text) > 80 else "")
        logger.info(f'Voice → Agent: "{preview}"')

        # ── Latency telemetry per turn ──
        # `t0_in` is the moment STT delivered the final transcript and
        # the LLM call is about to start. STT/VAD endpointing happens
        # before this in the LiveKit plugin layer — we time it from the
        # `on_user_speech_committed` hook in voice/agent.py (logged
        # separately as "stt_endpoint_ms" on the session record).
        t0_in = time.perf_counter()
        first_token_t: float | None = None
        first_token_ms: int | None = None

        # Open a Langfuse span around the agent turn so the whole
        # STT-complete → first-token → done timeline is visible in
        # one trace. Uses the v4 SDK API; silently no-ops on older
        # SDKs or when Langfuse isn't configured.
        lf_span_ctx = None
        lf_span = None
        lf_client = None
        try:
            from infrastructure.observability import get_langfuse
            lf_client = get_langfuse()
            if lf_client is not None and hasattr(lf_client, "start_as_current_observation"):
                lf_span_ctx = lf_client.start_as_current_observation(
                    as_type="span",
                    name="voice_pipeline",
                    input={"transcript": user_text},
                    metadata={
                        "user_id": self._user_id,
                        "session_id": self._session_id,
                        "stage": "agent_turn",
                        "path": "voice_fast",
                    },
                )
                lf_span = lf_span_ctx.__enter__()
                # Tag the trace for filtering in the Langfuse UI. v4
                # uses set_current_trace_io for trace input/output and
                # update_current_span for trace-level attributes.
                if hasattr(lf_client, "set_current_trace_io"):
                    lf_client.set_current_trace_io(input={"transcript": user_text})
        except Exception as e:
            logger.debug(f"Langfuse span attach failed (non-fatal): {e}")
            lf_span = None
            lf_span_ctx = None

        try:
            self._adapter._current_task = asyncio.current_task()

            # ── Stream tokens from the orchestrator into TTS ──
            chunk_idx = 0
            full_answer_parts: list[str] = []
            spoken_partial = ""    # running concatenation; truth on barge-in
            final_response = None

            # Voice uses the FAST path: single streaming LLM call, no
            # multi-agent fan-out, no router LLM, memory writes in the
            # background. See AgentOrchestrator.achat_stream_fast.
            #
            # On barge-in (LiveKit fires CancelledError into this task),
            # the async-for terminates without a "final" event. The
            # orchestrator itself catches the cancellation and saves the
            # partial answer to memory — we just need to stop forwarding
            # tokens to TTS.
            async for kind, payload in self._orchestrator.achat_stream_fast(
                user_message=user_text,
                user_id=self._user_id,
                session_id=self._session_id,
            ):
                if kind == "token":
                    if first_token_t is None:
                        first_token_t = time.perf_counter()
                        first_token_ms = int((first_token_t - t0_in) * 1000)
                        logger.info(f"⏱  first LLM token in {first_token_ms} ms")
                    full_answer_parts.append(payload)
                    self._event_ch.send_nowait(
                        ChatChunk(
                            id=f"lg-{chunk_idx}",
                            delta=ChoiceDelta(role="assistant", content=payload),
                        )
                    )
                    chunk_idx += 1
                elif kind == "partial":
                    # Snapshot of what's been emitted so far. Used by
                    # the cancellation path below as the "truth" of
                    # what was spoken in case of barge-in.
                    spoken_partial = payload
                elif kind == "final":
                    final_response = payload

            t_done = time.perf_counter()
            total_ms = int((t_done - t0_in) * 1000)
            llm_total_ms = (
                int((t_done - first_token_t) * 1000) if first_token_t else None
            )
            full_answer = "".join(full_answer_parts).strip()
            ans_preview = full_answer[:80] + ("..." if len(full_answer) > 80 else "")
            route = getattr(final_response, "route", "?") if final_response else "?"

            # Detailed log line — one place to read latency from.
            logger.success(
                "📊 Voice turn timings: "
                f"first_token={first_token_ms}ms, "
                f"llm_total={llm_total_ms}ms, "
                f"agent_total={total_ms}ms, "
                f"chunks={chunk_idx}, route={route}\n"
                f"   answer: \"{ans_preview}\""
            )

            # Stash on the adapter so voice/agent.py can ship it to the
            # UI HUD over the LiveKit data channel + the next caller can
            # read it from logs.
            self._adapter.last_latency = {  # type: ignore[attr-defined]
                "first_token_ms": first_token_ms,
                "llm_total_ms": llm_total_ms,
                "agent_total_ms": total_ms,
                "chunks": chunk_idx,
                "answer_preview": ans_preview,
            }

            if lf_client is not None:
                try:
                    if hasattr(lf_client, "update_current_span"):
                        lf_client.update_current_span(
                            output={"answer": full_answer, "route": route},
                            metadata={
                                "first_token_ms": first_token_ms,
                                "llm_total_ms": llm_total_ms,
                                "agent_total_ms": total_ms,
                                "chunks": chunk_idx,
                            },
                        )
                    if hasattr(lf_client, "set_current_trace_io"):
                        lf_client.set_current_trace_io(
                            output={"answer": full_answer}
                        )
                except Exception:
                    pass

        except asyncio.CancelledError:
            elapsed = int((time.perf_counter() - t0_in) * 1000)
            preview = spoken_partial[:80] + ("..." if len(spoken_partial) > 80 else "")
            logger.info(
                f"🛑 Agent task cancelled after {elapsed} ms (barge-in). "
                f"Spoken so far: \"{preview}\" ({len(spoken_partial)} chars). "
                f"Orchestrator saved this partial to memory."
            )
            # Surface the partial to Langfuse before re-raising.
            if lf_client is not None:
                try:
                    if hasattr(lf_client, "update_current_span"):
                        lf_client.update_current_span(
                            output={"answer_partial": spoken_partial},
                            metadata={
                                "barge_in": True,
                                "agent_total_ms": elapsed,
                                "spoken_chars": len(spoken_partial),
                            },
                        )
                except Exception:
                    pass
            raise

        except Exception:
            logger.exception("Agent processing failed")
            # Don't leave TTS hanging — emit a graceful fallback so the
            # user hears *something* instead of dead air.
            self._event_ch.send_nowait(
                ChatChunk(
                    id="lg-error",
                    delta=ChoiceDelta(
                        role="assistant",
                        content=(
                            "I'm sorry, I had a problem processing that. "
                            "Could you please try again?"
                        ),
                    ),
                )
            )

        finally:
            self._adapter._current_task = None
            # Close the Langfuse span context if one was opened.
            if lf_span_ctx is not None:
                try:
                    lf_span_ctx.__exit__(None, None, None)
                except Exception:
                    pass
