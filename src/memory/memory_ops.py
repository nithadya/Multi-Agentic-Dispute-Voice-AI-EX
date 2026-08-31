"""
Memory operations — distill, recall, forget.

Three core operations on the memory system:
- Distill: Extract long-term facts from conversation turns (write path)
- Recall: Hybrid retrieval from short-term + long-term memory (read path)
- Forget: Soft deletion and decay/pruning (cleanup path)
"""

import json
import uuid
import time
from loguru import logger
from typing import List, Optional, Tuple

import tiktoken

from memory.schemas import ConversationTurn, MemoryFact
from memory.prompts import build_distill_prompt
from memory.policies import score_memory_fact, dedupe_facts
from infrastructure.observability import observe, update_current_observation


# ═══════════════════════════════════════════════════════════════════════════════
# Distiller — write path
# ═══════════════════════════════════════════════════════════════════════════════


class MemoryDistiller:
    """Extracts memorable facts from conversation turns."""

    def __init__(self, llm, lt_store):
        self.llm = llm
        self.lt_store = lt_store

    def should_distill(self, turns: List[ConversationTurn]) -> bool:
        """Check if distillation should be triggered."""
        if not turns:
            return False
        if len(turns) >= 5:
            return True
        keywords = ["remember", "from now on", "remind me", "always", "never"]
        for turn in turns:
            content_lower = turn.content.lower()
            if any(kw in content_lower for kw in keywords):
                return True
        return False

    @observe(name="distill_facts", as_type="generation")
    def distill(self, user_id: str, turns: List[ConversationTurn]) -> List[MemoryFact]:
        """
        Distill facts from conversation turns.

        Traced as a LangFuse **generation** to capture the distillation LLM cost.
        """
        if not turns:
            return []

        system_prompt, user_prompt = build_distill_prompt(turns)

        update_current_observation(
            input=user_prompt[:1000],
            model=self._model_name(),
        )

        try:
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
            content = response.content if hasattr(response, "content") else str(response)

            # Extract token usage if available
            usage = {}
            if hasattr(response, "response_metadata"):
                meta = response.response_metadata or {}
                token_usage = meta.get("token_usage") or meta.get("usage", {})
                if token_usage:
                    usage = {
                        "input": token_usage.get("prompt_tokens", 0),
                        "output": token_usage.get("completion_tokens", 0),
                        "total": token_usage.get("total_tokens", 0),
                    }

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            facts_data = json.loads(content)
        except Exception as e:
            logger.error(f"Failed to distill: {e}")
            return []

        now = time.time()
        facts = []
        for fact_data in facts_data:
            fact_id = str(uuid.uuid4())
            text = fact_data.get("text", "")
            tags = fact_data.get("tags", [])
            if not text:
                continue
            score = score_memory_fact(text=text, created_at=now, now=now, repetition_count=1)
            fact = MemoryFact(
                id=fact_id, user_id=user_id, text=text, score=score,
                tags=tags, created_at=now, last_used_at=now, ttl_at=None, pin=False,
            )
            facts.append(fact)

        embedder = self.lt_store.embedder if hasattr(self.lt_store, 'embedder') else None
        facts = dedupe_facts(facts, embedder=embedder)
        if facts:
            self.lt_store.upsert(facts)
            logger.info(f"Distilled {len(facts)} facts for user {user_id}")

        update_current_observation(
            output=f"Distilled {len(facts)} facts",
            usage=usage if usage else None,
            metadata={"facts_count": len(facts), "user_id": user_id},
        )

        return facts

    def _model_name(self) -> str:
        """Extract model name from the LLM for LangFuse metadata."""
        llm = getattr(self.llm, "runnable", self.llm)
        if hasattr(llm, "model_name"):
            return llm.model_name
        if hasattr(llm, "model"):
            return llm.model
        return "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# Recaller — read path
# ═══════════════════════════════════════════════════════════════════════════════


class MemoryRecaller:
    """Combines short-term and long-term memory with token budget."""

    def __init__(self, st_store, lt_store):
        self.st_store = st_store
        self.lt_store = lt_store
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    @observe(name="memory_recall_inner")
    def recall(
        self, user_id: str, session_id: str, query: str,
        k_st: int = 6, k_lt: int = 5, max_tokens: int = 500,
    ) -> Tuple[List[ConversationTurn], List[MemoryFact]]:
        """Recall relevant memories (traced via LangFuse)."""
        from infrastructure.config import LT_SIM_THRESHOLD

        st_turns = self.st_store.recent(user_id, session_id, k_st)
        lt_facts = self.lt_store.query(
            user_id=user_id, query_text=query, k=k_lt, threshold=LT_SIM_THRESHOLD,
        )

        st_turns_filtered, lt_facts_filtered = self._budget_tokens(st_turns, lt_facts, max_tokens)

        update_current_observation(
            input=query,
            metadata={
                "st_raw": len(st_turns),
                "st_filtered": len(st_turns_filtered),
                "lt_raw": len(lt_facts),
                "lt_filtered": len(lt_facts_filtered),
            },
        )

        logger.info(
            f"Recalled {len(st_turns_filtered)} ST turns, "
            f"{len(lt_facts_filtered)} LT facts for user {user_id}"
        )
        return st_turns_filtered, lt_facts_filtered

    def _budget_tokens(
        self, st_turns: List[ConversationTurn], lt_facts: List[MemoryFact], max_tokens: int,
    ) -> Tuple[List[ConversationTurn], List[MemoryFact]]:
        """Apply token budget to memories."""
        st_budget = int(max_tokens * 0.6)
        lt_budget = int(max_tokens * 0.4)

        st_filtered = []
        st_tokens = 0
        for turn in reversed(st_turns):
            turn_tokens = self.count_tokens(turn.content)
            if st_tokens + turn_tokens <= st_budget:
                st_filtered.insert(0, turn)
                st_tokens += turn_tokens
            if len(st_filtered) >= 4:
                break

        lt_filtered = []
        lt_tokens = 0
        for fact in lt_facts:
            fact_tokens = self.count_tokens(fact.text)
            if lt_tokens + fact_tokens <= lt_budget:
                lt_filtered.append(fact)
                lt_tokens += fact_tokens
            if len(lt_filtered) >= 3:
                break

        logger.debug(f"Token budget: ST={st_tokens}/{st_budget}, LT={lt_tokens}/{lt_budget}")
        return st_filtered, lt_filtered

    def format_context(self, st_turns: List[ConversationTurn]) -> str:
        """Format recalled memories as context string."""
        lines = []
        if st_turns:
            lines.append("=== RECENT CONVERSATION ===")
            for turn in st_turns:
                role = turn.role.capitalize()
                lines.append(f"{role}: {turn.content}")
            lines.append("")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Forget service — cleanup path
# ═══════════════════════════════════════════════════════════════════════════════


class MemoryForgetService:
    """Handles soft deletion and decay/pruning."""

    def __init__(self, lt_store):
        self.lt_store = lt_store

    def forget(self, user_id: str, fact_id: str) -> None:
        """Soft delete a memory fact."""
        self.lt_store.soft_delete(user_id, fact_id)
        logger.info(f"Forgot fact {fact_id} for user {user_id}")

    def decay_and_prune(self, now: Optional[float] = None) -> int:
        """Apply decay and prune expired facts."""
        if now is None:
            now = time.time()
        pruned = self.lt_store.decay_and_prune(now)
        logger.info(f"Pruned {pruned} facts at timestamp {now}")
        return pruned
