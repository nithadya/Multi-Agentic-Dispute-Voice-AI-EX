"""
Memory prompts — distillation and recall prompt templates.

Prompts are fetched from **LangFuse Prompt Management** at runtime.
Local fallbacks below are used when the prompt hasn't been created
in LangFuse yet, so the system works out-of-the-box.

To manage these prompts in LangFuse Cloud:
  1. Open LangFuse → Prompts → + New Prompt
  2. Create prompts with the names shown in LANGFUSE_PROMPT_NAMES
  3. Use {{variable}} (Mustache syntax) for template variables
  4. Publish a version → it's live instantly, no code deploy needed
"""

from infrastructure.observability import fetch_prompt

# ─────────────────────────────────────────────────────────────
# LangFuse prompt names → create these in your dashboard
# ─────────────────────────────────────────────────────────────

LANGFUSE_PROMPT_NAMES = {
    "distill_system": "disputeai-distill-system",
    "distill_user":   "disputeai-distill-user",
    "recall_system":  "disputeai-recall-system",
    "recall_user":    "disputeai-recall-user",
}

# ─────────────────────────────────────────────────────────────
# Fallback: Distillation prompts
# ─────────────────────────────────────────────────────────────

_DISTILL_SYSTEM_FALLBACK = """\
You are a memory extraction specialist for an E-Commerce Dispute Resolution AI.

Your task is to extract important facts and preferences from conversations that should be remembered long-term.

EXTRACTION RULES:
1. Extract explicit customer preferences, habits, and instructions
2. Extract facts mentioned multiple times (indicates importance)
3. Extract instructions prefixed with "remember", "always", "never", "from now on"
4. Extract reminder requests with timing information
5. Skip casual chitchat and one-time situational details

AUTOMATIC CATEGORIZATION:
Automatically determine the appropriate tags/categories for each fact. Common e-commerce dispute categories include:
- order, dispute, refund, return, tracking
- vendor, product, category, warranty
- fraud, escalation, evidence
- preference, communication_preference
- loyalty_tier, account, profile
- reminder, follow_up, task
- payment, billing, chargeback

OUTPUT FORMAT:
Return a JSON array of facts. Each fact should have:
{
  "text": "The distilled fact in natural language (e.g., 'Customer has an open dispute DIS-001 for a damaged laptop')",
  "tags": ["dispute", "damaged", "order"],  // Auto-detected categories (2-4 tags per fact)
  "has_reminder": false,  // true if this is a reminder request
  "time_info": null  // timing details if has_reminder is true (e.g., "daily at 6am", "every Monday")
}

IMPORTANT:
- Be concise. One fact per important item
- Maximum 10 facts per extraction
- Always include 2-4 relevant tags per fact
- Extract customer name if mentioned for personalization

Example output:
[
  {
    "text": "Customer has an open return dispute for Order ORD-1045 (damaged smartphone screen)",
    "tags": ["dispute", "return", "damaged", "order"],
    "has_reminder": false,
    "time_info": null
  },
  {
    "text": "Customer prefers email updates over SMS for dispute status",
    "tags": ["preference", "communication_preference"],
    "has_reminder": false,
    "time_info": null
  },
  {
    "text": "Remind customer to submit photos of damaged item by Friday",
    "tags": ["reminder", "evidence", "dispute"],
    "has_reminder": true,
    "time_info": "Friday"
  }
]"""

_DISTILL_USER_FALLBACK = """\
Extract memorable facts from this conversation:

{conversation}

Return JSON array of facts:"""

# ─────────────────────────────────────────────────────────────
# Fallback: Recall prompts
# ─────────────────────────────────────────────────────────────

_RECALL_SYSTEM_FALLBACK = """\
You are a memory recall assistant.

You help retrieve relevant memories based on the current conversation context.

RECALL RULES:
1. Prioritize memories that directly relate to the current query
2. Include both recent context (short-term) and relevant facts (long-term)
3. Keep total context under 500 tokens
4. Format memories clearly with timestamps
5. Distinguish between ST (short-term, conversational) and LT (long-term, factual)

OUTPUT FORMAT:
Return a formatted memory context that can be injected into a chat prompt."""

_RECALL_USER_FALLBACK = """\
Retrieve and format memories for this query:

QUERY: {query}

SHORT-TERM CONTEXT (recent conversation):
{st_context}

LONG-TERM FACTS (distilled knowledge):
{lt_facts}

Format a concise memory context (≤500 tokens):"""


# ─────────────────────────────────────────────────────────────
# Prompt builders — fetch from LangFuse, fall back to local
# ─────────────────────────────────────────────────────────────


def build_distill_prompt(turns: list) -> tuple[str, str]:
    """Build complete distillation prompt (LangFuse → local fallback)."""
    conversation = format_conversation_for_distill(turns)

    system_prompt = fetch_prompt(
        LANGFUSE_PROMPT_NAMES["distill_system"],
        fallback=_DISTILL_SYSTEM_FALLBACK,
    )
    user_prompt = fetch_prompt(
        LANGFUSE_PROMPT_NAMES["distill_user"],
        fallback=_DISTILL_USER_FALLBACK,
        conversation=conversation,
    )
    return system_prompt, user_prompt


def build_recall_prompt(
    query: str, st_turns: list, lt_facts: list
) -> tuple[str, str]:
    """Build complete recall prompt (LangFuse → local fallback)."""
    st_context = format_st_context(st_turns)
    lt_context = format_lt_facts(lt_facts)

    system_prompt = fetch_prompt(
        LANGFUSE_PROMPT_NAMES["recall_system"],
        fallback=_RECALL_SYSTEM_FALLBACK,
    )
    user_prompt = fetch_prompt(
        LANGFUSE_PROMPT_NAMES["recall_user"],
        fallback=_RECALL_USER_FALLBACK,
        query=query,
        st_context=st_context,
        lt_facts=lt_context,
    )
    return system_prompt, user_prompt


# ─────────────────────────────────────────────────────────────
# Formatting helpers (unchanged)
# ─────────────────────────────────────────────────────────────


def format_conversation_for_distill(turns: list) -> str:
    """Format conversation turns for distillation prompt."""
    lines = []
    for turn in turns:
        role = turn.role.capitalize()
        lines.append(f"{role}: {turn.content}")
    return "\n".join(lines)


def format_st_context(turns: list) -> str:
    """Format short-term context for recall."""
    if not turns:
        return "(No recent context)"

    lines = []
    for turn in turns:
        role = turn.role.capitalize()
        content = turn.content[:200] + "..." if len(turn.content) > 200 else turn.content
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)


def format_lt_facts(facts: list) -> str:
    """Format long-term facts for recall."""
    if not facts:
        return "(No long-term facts)"

    lines = []
    for i, fact in enumerate(facts, 1):
        tags_str = f"[{', '.join(fact.tags)}]" if fact.tags else ""
        lines.append(f"{i}. {fact.text} {tags_str} (score: {fact.score:.2f})")
    return "\n".join(lines)


def format_procedures(procedures: list) -> str:
    """Format procedural memory (workflows) for agent context."""
    if not procedures:
        return "(No relevant procedures found)"

    lines = []
    for i, proc in enumerate(procedures, 1):
        lines.append(f"\n**Procedure {i}: {proc.name}** ({proc.category})")
        lines.append(f"Description: {proc.description}")

        if proc.context_when:
            lines.append(f"When to use: {proc.context_when}")

        lines.append("\nSteps:")
        for step in proc.steps:
            order = step.get("order", "")
            action = step.get("action", "")
            desc = step.get("description", "")
            if action and desc:
                lines.append(f"  {order}. {action}: {desc}")
            elif desc:
                lines.append(f"  {order}. {desc}")

        if proc.conditions:
            lines.append(f"\nConditions: {proc.conditions}")

        lines.append("")  # Blank line between procedures

    return "\n".join(lines)
