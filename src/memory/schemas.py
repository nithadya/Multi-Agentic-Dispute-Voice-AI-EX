"""
Memory schemas and interfaces.

Dataclasses for conversations, facts, episodes, procedures, and reminder intents.
Protocol definitions for stores, embedder, and clock contracts.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any, Protocol, Iterable


@dataclass
class ConversationTurn:
    """
    A single turn in a conversation (user or assistant message).
    
    Stored in short-term memory (Supabase ``st_turns`` table) as a ring buffer.
    """
    user_id: str
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    ts: float  # epoch seconds
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "ts": self.ts,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ConversationTurn":
        """Create from dictionary."""
        return cls(
            user_id=data["user_id"],
            session_id=data["session_id"],
            role=data["role"],
            content=data["content"],
            ts=data["ts"],
        )


@dataclass
class MemoryFact:
    """
    A distilled long-term memory fact extracted from conversations.
    
    Stored in Supabase Postgres (metadata + pgvector embeddings).
    """
    id: str
    user_id: str
    text: str
    score: float  # 0.0-1.0, composite of recency + repetition + explicitness
    tags: List[str] = field(default_factory=list)
    created_at: float = 0.0  # epoch seconds
    last_used_at: float = 0.0  # epoch seconds
    ttl_at: Optional[float] = None  # epoch seconds, None = no expiry
    pin: bool = False  # if True, do not shift reminders on collisions
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "text": self.text,
            "score": self.score,
            "tags": self.tags,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "ttl_at": self.ttl_at,
            "pin": self.pin,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "MemoryFact":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            text=data["text"],
            score=data["score"],
            tags=data.get("tags", []),
            created_at=data.get("created_at", 0.0),
            last_used_at=data.get("last_used_at", 0.0),
            ttl_at=data.get("ttl_at"),
            pin=data.get("pin", False),
        )


@dataclass
class Episode:
    """
    A conversation episode stored in episodic long-term memory.
    
    Represents a complete conversation session with full context.
    Stored in Supabase Postgres (metadata + pgvector summary embedding).
    """
    id: str
    user_id: str
    session_id: str
    turns: List[ConversationTurn]
    summary: str  # Brief summary of the conversation
    topic_tags: List[str] = field(default_factory=list)
    start_at: float = 0.0  # First turn timestamp
    end_at: float = 0.0  # Last turn timestamp
    turn_count: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "turns": [turn.to_dict() for turn in self.turns],
            "summary": self.summary,
            "topic_tags": self.topic_tags,
            "start_at": self.start_at,
            "end_at": self.end_at,
            "turn_count": self.turn_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Episode":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            session_id=data["session_id"],
            turns=[ConversationTurn.from_dict(t) for t in data["turns"]],
            summary=data["summary"],
            topic_tags=data.get("topic_tags", []),
            start_at=data.get("start_at", 0.0),
            end_at=data.get("end_at", 0.0),
            turn_count=data.get("turn_count", 0),
        )


@dataclass
class ReminderIntent:
    """
    A reminder intent extracted from a memory fact.
    
    Can specify RRULE (recurring) or offset (one-time) reminders.
    """
    fact_id: str
    user_id: str
    title: str
    rrule: Optional[str] = None  # e.g., "FREQ=DAILY;BYHOUR=8;BYMINUTE=0"
    offset: Optional[Dict] = None  # e.g., {"minutes": 45}, {"hours": 2}
    meta: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "fact_id": self.fact_id,
            "user_id": self.user_id,
            "title": self.title,
            "rrule": self.rrule,
            "offset": self.offset,
            "meta": self.meta or {},
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ReminderIntent":
        """Create from dictionary."""
        return cls(
            fact_id=data["fact_id"],
            user_id=data["user_id"],
            title=data["title"],
            rrule=data.get("rrule"),
            offset=data.get("offset"),
            meta=data.get("meta"),
        )


@dataclass
class Procedure:
    """
    A procedural memory - step-by-step workflow for task execution.
    
    Stored in Supabase Postgres with pgvector embeddings for semantic retrieval.
    Represents "how-to" knowledge that guides the agent through multi-step tasks.
    """
    id: str
    name: str
    description: str
    steps: List[Dict[str, Any]]  # Ordered list of steps
    context_when: Optional[str] = None
    conditions: Optional[Dict[str, Any]] = None
    examples: Optional[List[str]] = field(default_factory=list)
    category: Optional[str] = None
    similarity: Optional[float] = None  # Populated during retrieval
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "context_when": self.context_when,
            "conditions": self.conditions,
            "examples": self.examples,
            "category": self.category,
            "similarity": self.similarity,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Procedure":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            steps=data["steps"],
            context_when=data.get("context_when"),
            conditions=data.get("conditions"),
            examples=data.get("examples", []),
            category=data.get("category"),
            similarity=data.get("similarity"),
        )
    
    def format_steps(self) -> str:
        """
        Format steps as a numbered list for display to LLM or user.
        
        Returns:
            Human-readable step-by-step guide
        """
        if not self.steps:
            return "No steps defined."
        
        lines = [f"**{self.name}**: {self.description}", ""]
        
        if self.context_when:
            lines.append(f"**When to use**: {self.context_when}")
            lines.append("")
        
        lines.append("**Steps**:")
        for i, step in enumerate(self.steps, 1):
            action = step.get("action", "")
            desc = step.get("description", "")
            if action and desc:
                lines.append(f"{i}. **{action}**: {desc}")
            elif desc:
                lines.append(f"{i}. {desc}")
            else:
                lines.append(f"{i}. {action}")
        
        if self.conditions:
            lines.append("")
            lines.append(f"**Conditions**: {self.conditions}")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Protocol interfaces
# ═══════════════════════════════════════════════════════════════════════════════


class ShortTermStore(Protocol):
    """
    Short-term memory store (Supabase ``st_turns`` table).

    Stores recent conversation turns as a ring buffer with TTL.
    """

    def append(
        self,
        turn: ConversationTurn,
        max_turns: int,
        ttl_seconds: int,
    ) -> None:
        ...

    def recent(
        self,
        user_id: str,
        session_id: str,
        k: int,
    ) -> List[ConversationTurn]:
        ...


class LongTermStore(Protocol):
    """
    Long-term memory store (Supabase Postgres + pgvector).

    Stores distilled facts with metadata and vector embeddings.
    """

    def upsert(self, facts: Iterable[MemoryFact]) -> None:
        ...

    def query(
        self,
        user_id: str,
        text: str,
        k: int,
        threshold: float,
    ) -> List[MemoryFact]:
        ...

    def soft_delete(self, user_id: str, fact_id: str) -> None:
        ...

    def decay_and_prune(self, now: float) -> int:
        ...

    def list_reminder_intents(self, user_id: str) -> List[ReminderIntent]:
        ...


class Embedder(Protocol):
    """Text embedder protocol — wraps embedding model for consistent interface."""

    def embed(self, texts: List[str]) -> List[List[float]]:
        ...


class Clock(Protocol):
    """Clock abstraction — allows fast-forwarding time in tests."""

    def now(self) -> float:
        ...
