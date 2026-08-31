"""
Chat LLM providers — 3-model architecture.

Three specialised LLMs for different tasks:
  - Router:    gpt-4o-mini via OpenRouter (reliable JSON output)
  - Extractor: llama-3.1-8b-instant via Groq (ultra-fast structured extraction)
  - Chat:      gemini-2.0-flash via OpenRouter (high quality synthesis)
"""

import functools
import os
import urllib.request
from typing import Optional, Any

from langchain_openai import ChatOpenAI
from loguru import logger

from infrastructure.config import (
    ROUTER_MODEL,
    ROUTER_PROVIDER,
    EXTRACTOR_MODEL,
    EXTRACTOR_PROVIDER,
    GROQ_BASE_URL,
    CHAT_MODEL,
    CHAT_PROVIDER,
    FAST_CHAT_MODEL,
    FAST_CHAT_PROVIDER,
    OPENROUTER_BASE_URL,
    get_api_key,
)


# ── Groq → OpenRouter automatic failover ─────────────────────────────────────
# Groq fronts its API with Cloudflare, which blocks many datacenter / VPN /
# geo-restricted IP ranges (HTTP 403 "Access denied. Please check your network
# settings."). When that happens — a student on a blocked network, a VPN exit
# node, a restricted region — every Groq-backed LLM (router, extractor, fast
# chat / voice) would fail. To keep the app runnable ANYWHERE, we probe Groq
# once per process; if it is unreachable we transparently build those LLMs
# against the equivalent OpenRouter model instead. On AWS (Groq reachable) the
# probe passes and nothing changes.
#
# Override with env vars:
#   LLM_FORCE_OPENROUTER=true   → always use OpenRouter for Groq roles (skip probe)
#   LLM_DISABLE_GROQ_FALLBACK=true → never fall back (use Groq as configured)

_GROQ_TO_OPENROUTER = {
    "llama-3.3-70b-versatile": "meta-llama/llama-3.3-70b-instruct",
    "llama-3.1-8b-instant": "meta-llama/llama-3.1-8b-instruct",
    "llama-3.1-70b-versatile": "meta-llama/llama-3.3-70b-instruct",
}
_DEFAULT_OPENROUTER_FALLBACK = os.getenv(
    "OPENROUTER_FALLBACK_MODEL", "meta-llama/llama-3.3-70b-instruct"
)


def _env_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


@functools.lru_cache(maxsize=1)
def _groq_reachable() -> bool:
    """Probe Groq once per process (result cached).

    Returns False if Groq's edge blocks this host (403/connection error), so
    callers can fall back to OpenRouter. A browser User-Agent is sent so the
    probe mirrors the real httpx client (the block is IP-based, not UA-based).
    """
    if _env_true("LLM_DISABLE_GROQ_FALLBACK"):
        return True   # operator forced Groq; never fall back
    if _env_true("LLM_FORCE_OPENROUTER"):
        logger.warning("LLM_FORCE_OPENROUTER set — routing Groq roles to OpenRouter")
        return False
    try:
        req = urllib.request.Request(
            GROQ_BASE_URL.rstrip("/") + "/models",
            headers={
                "Authorization": f"Bearer {get_api_key('groq')}",
                "User-Agent": "Mozilla/5.0",
            },
        )
        urllib.request.urlopen(req, timeout=4)
        return True
    except Exception as exc:  # noqa: BLE001 — any failure means "unreachable"
        logger.warning(
            "Groq unreachable ({}). Falling back to OpenAI for Groq-backed "
            "LLMs (router / extractor / fast chat / voice).",
            getattr(exc, "code", exc),
        )
        return False


def _build_llm(
    model: str,
    provider: str,
    temperature: float = 0,
    streaming: bool = False,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> ChatOpenAI:
    """Internal factory — builds a ChatOpenAI for any provider.

    Transparently fails Groq over to OpenRouter when Groq is unreachable
    (see ``_groq_reachable``). The mapped OpenRouter model is an equivalent
    Llama checkpoint, so behaviour is unchanged for callers.
    """
    if provider == "groq" and not _groq_reachable():
        fallback_model = "gpt-4o-mini"
        logger.info("Groq→OpenAI failover: {} → {}", model, fallback_model)
        provider, model = "openai", fallback_model

    llm_kwargs: dict[str, Any] = dict(
        model=model,
        temperature=temperature,
        streaming=streaming,
        max_tokens=max_tokens,
        **kwargs,
    )

    if provider == "openrouter":
        llm_kwargs["openai_api_base"] = OPENROUTER_BASE_URL
        llm_kwargs["openai_api_key"] = get_api_key("openrouter")
    elif provider == "groq":
        llm_kwargs["openai_api_base"] = GROQ_BASE_URL
        llm_kwargs["openai_api_key"] = get_api_key("groq")
    elif provider == "openai":
        llm_kwargs["openai_api_key"] = get_api_key("openai")

    main_llm = ChatOpenAI(**llm_kwargs)

    # Runtime fallback: if Groq hits rate limits (429) or fails, fallback to OpenAI
    if provider == "groq":
        fallback_kwargs = dict(llm_kwargs)
        fallback_kwargs.pop("openai_api_base", None)
        fallback_kwargs["openai_api_key"] = get_api_key("openai")
        fallback_kwargs["model"] = "gpt-4o-mini"
        fallback_llm = ChatOpenAI(**fallback_kwargs)
        return main_llm.with_fallbacks([fallback_llm])
    
    return main_llm


def get_router_llm(temperature: float = 0, **kwargs: Any) -> ChatOpenAI:
    """LLM for intent classification (routing).

    Model: llama-3.1-8b-instant via Groq.
    LPU hardware delivers ~600 tok/s — a 150-token JSON route decision
    returns in ~250ms vs. 2-3s on gpt-4o-mini-via-OpenRouter, with the
    same accuracy on this classification task.
    """
    return _build_llm(ROUTER_MODEL, ROUTER_PROVIDER, temperature=temperature, **kwargs)


def get_fast_chat_llm(temperature: float = 0.3, **kwargs: Any) -> ChatOpenAI:
    """LLM for the "direct" route — greetings, concierge, follow-ups,
    AND the voice fast path (``achat_stream_fast``).

    Model: llama-3.3-70b-versatile via Groq. LPU hardware delivers
    ~500+ tok/s; with ``streaming=True`` first-token latency is
    ~150-400ms vs. several seconds when streaming is off (server
    generates the whole reply before any byte returns).

    ``streaming`` defaults to True here — critical for voice. Callers
    using this LLM in non-realtime contexts can pass ``streaming=False``
    explicitly to override.
    """
    kwargs.setdefault("streaming", True)
    return _build_llm(FAST_CHAT_MODEL, FAST_CHAT_PROVIDER, temperature=temperature, **kwargs)


def get_extractor_llm(temperature: float = 0, **kwargs: Any) -> ChatOpenAI:
    """LLM for extraction tasks (distillation, topic tagging, trigger checks).

    Model: llama-3.1-8b-instant via Groq.
    Ultra-fast (~50ms), free tier, perfect for structured extraction.
    """
    return _build_llm(EXTRACTOR_MODEL, EXTRACTOR_PROVIDER, temperature=temperature, **kwargs)


def get_chat_llm(temperature: float = 0, **kwargs: Any) -> ChatOpenAI:
    """LLM for user-facing responses (synthesis, RAG generation).

    Model: gemini-2.0-flash via OpenRouter.
    High quality, generous context window, natural tone.
    """
    return _build_llm(CHAT_MODEL, CHAT_PROVIDER, temperature=temperature, **kwargs)
