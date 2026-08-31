"""
Domain Guardrail — keeps the assistant on-topic.

A binary classifier that decides whether the user message is within
the scope of a DisputeAI health assistant. Runs on Llama
3.1 8B Instant via Groq (~150 ms) and is invoked in parallel with
the router / CAG / memory recall, so its latency is hidden inside
the existing parallel batch.

When ``out_of_scope`` is returned, the chat pipeline short-circuits
to a templated polite refusal — no router classification, no tool
call, no synthesis LLM call. This is what stops the bot from
answering "who is the president of the USA?" or burning a Tavily
search on unrelated weather questions.

The guardrail fails *open* on any error: a transient Groq outage
should not block legitimate users, so we let the request continue
through the normal router path.
"""

from __future__ import annotations

from typing import Any, Literal
from loguru import logger

from infrastructure.observability import observe, update_current_observation


GuardrailVerdict = Literal["in_scope", "out_of_scope"]


_GUARDRAIL_SYSTEM = """\
You are a scope filter for an E-Commerce Dispute Resolution AI assistant.

Decide whether the user's message is within the assistant's domain.

IN-SCOPE — the assistant should help with:
  • Order status, tracking, and lookup
  • Dispute submissions, returns, refunds, partial refunds, replacement requests
  • Damaged items, missing orders, fraud, warranty claims, chargeback questions
  • Policies on return windows, refund eligibility, evidence requirements
  • Customer chitchat, greetings, small talk, thanks
  • User identity info provided on request (e.g. raw phone numbers, emails, order IDs)
OUT-OF-SCOPE — politely refuse:
  • Healthcare, medical advice, doctors, appointments, hospital queries (these are unrelated!)
  • General world knowledge (presidents, capitals, sports, history, trivia)
  • Coding help, math problems, jokes, riddles, role-play
  • Gibberish or random strings (BUT raw phone numbers like +94781030736 or 94781030736 are IN SCOPE)
  • Anything unrelated to an e-commerce order or vendor policy dispute

Answer with ONE WORD ONLY: ``in_scope`` or ``out_of_scope``.
No explanation, no punctuation, no other tokens.
"""


# Few-shot examples baked into the user-prompt template — keeps the
# 8B model honest without burning a separate fine-tune.
_GUARDRAIL_EXAMPLES = """\
Examples:
  USER: "Can I return a laptop after 14 days?"      → in_scope
  USER: "Check status of dispute DIS-2024-001"     → in_scope
  USER: "94781030736"                              → in_scope
  USER: "+94781030736"                             → in_scope
  USER: "Hey there"                                 → in_scope
  USER: "Where is my refund for order 123?"         → in_scope
  USER: "Who is the cardiologist?"                 → out_of_scope
  USER: "Who is the president of the USA?"          → out_of_scope
  USER: "What's the capital of France?"             → out_of_scope
  USER: "Write me a Python function"                → out_of_scope
  USER: "How do I cure a headache?"                 → out_of_scope
"""


def _build_user_prompt(message: str) -> str:
    return f"{_GUARDRAIL_EXAMPLES}\n\nUSER: \"{(message or '').strip()}\"\n→"


class Guardrail:
    """Binary in_scope / out_of_scope classifier."""

    def __init__(self, llm: Any) -> None:
        """
        Args:
            llm: A LangChain ``ChatOpenAI``-compatible instance.
                Use the extractor LLM (Llama 3.1 8B on Groq) — it's
                cheap and fast enough that parallel guardrail latency
                is hidden behind the router (~800 ms) in every gather.
        """
        self.llm = llm

    @observe(name="guardrail", as_type="generation")
    async def aclassify(self, message: str) -> GuardrailVerdict:
        """Classify *message* as ``in_scope`` or ``out_of_scope``.

        Fails open: any LLM error returns ``in_scope`` so transient
        provider issues don't lock real users out of the assistant.
        """
        msgs = [
            {"role": "system", "content": _GUARDRAIL_SYSTEM},
            {"role": "user", "content": _build_user_prompt(message)},
        ]
        try:
            response = await self.llm.ainvoke(msgs)
        except Exception as exc:
            logger.warning("Guardrail LLM error (failing open): {}", exc)
            return "in_scope"

        raw = (
            response.content if hasattr(response, "content") else str(response)
        ).strip().lower()

        # Be permissive in parsing — the model occasionally adds quotes,
        # backticks, or trailing punctuation despite the instruction.
        verdict: GuardrailVerdict
        if "out_of_scope" in raw or "out-of-scope" in raw or "out of scope" in raw:
            verdict = "out_of_scope"
        elif "in_scope" in raw or "in-scope" in raw or "in scope" in raw:
            verdict = "in_scope"
        else:
            # Unrecognised response — safest default is to let the
            # normal pipeline handle it.
            logger.debug("Guardrail unparsable response {!r} → defaulting in_scope", raw[:50])
            verdict = "in_scope"

        update_current_observation(
            input=(message or "")[:200],
            output=verdict,
        )
        return verdict


# Templated refusal returned when the guardrail says out_of_scope.
# Kept short and friendly — the bot is a store concierge, not a
# rule enforcer. Single source of truth so we only edit it here.
OUT_OF_SCOPE_REPLY = (
    "I'm the DisputeAI E-Commerce Specialist — I can help with your "
    "orders, return claims, store policies, and refund calculations. "
    "That's outside what I'm built for, but I'm happy to help with anything "
    "related to your disputes or orders. What can I do for you?"
)
