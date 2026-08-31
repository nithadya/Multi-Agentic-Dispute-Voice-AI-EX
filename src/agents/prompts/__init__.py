"""
Agent prompt templates — router, synthesiser, system persona.

Prompts are fetched from LangFuse Prompt Management at runtime.
Local fallbacks are defined in ``agent_prompts.py``.
"""

from .agent_prompts import (
    LANGFUSE_PROMPT_NAMES,
    build_router_prompt,
    build_synthesiser_prompt,
)

__all__ = [
    "LANGFUSE_PROMPT_NAMES",
    "build_router_prompt",
    "build_synthesiser_prompt",
]
