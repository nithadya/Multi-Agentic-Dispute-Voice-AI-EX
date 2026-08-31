"""
Memory policies — scoring, decay, pruning, deduplication.

Pure business rules for memory lifecycle management (no I/O).
"""

import math
from typing import List
from .schemas import MemoryFact


def score_memory_fact(
    text: str,
    created_at: float,
    now: float,
    repetition_count: int = 1,
    explicit_keywords: List[str] = None,
) -> float:
    """
    Calculate memory fact score based on recency, repetition, and explicitness.
    
    Score = 0.5 * recency + 0.3 * repetition + 0.2 * explicitness
    """
    if explicit_keywords is None:
        explicit_keywords = ["remember", "always", "never", "from now on", "remind"]
    
    # Recency component (exponential decay with 7-day half-life)
    age_days = (now - created_at) / 86400
    recency_score = math.exp(-age_days / 7.0)
    
    # Repetition component (logarithmic scaling, caps at 10 mentions)
    repetition_score = min(1.0, math.log(1 + repetition_count) / math.log(11))
    
    # Explicitness component (keyword presence)
    text_lower = text.lower()
    explicit_matches = sum(1 for kw in explicit_keywords if kw in text_lower)
    explicitness_score = min(1.0, explicit_matches / len(explicit_keywords))
    
    # Composite score
    score = 0.5 * recency_score + 0.3 * repetition_score + 0.2 * explicitness_score
    
    return round(score, 3)


def apply_decay(fact: MemoryFact, now: float, half_life_days: float) -> float:
    """
    Apply decay to memory fact score based on age.
    
    Uses exponential decay with configurable half-life.
    """
    age_days = (now - fact.last_used_at) / 86400
    decay_factor = math.exp(-age_days * math.log(2) / half_life_days)
    
    return round(fact.score * decay_factor, 3)


def should_prune(fact: MemoryFact, now: float, ttl_seconds: int, min_score: float = 0.1) -> bool:
    """
    Determine if a memory fact should be pruned.
    
    Prunes if:
    - TTL expired (if set)
    - Score below minimum threshold
    - Pinned facts are never pruned
    """
    # Never prune pinned facts
    if fact.pin:
        return False
    
    # Check TTL
    if fact.ttl_at is not None and now >= fact.ttl_at:
        return True
    
    # Check default TTL (90 days from creation)
    default_expiry = fact.created_at + ttl_seconds
    if now >= default_expiry:
        return True
    
    # Check score threshold
    if fact.score < min_score:
        return True
    
    return False


def dedupe_facts(
    facts: List[MemoryFact],
    similarity_threshold: float = 0.85,
    embedder=None,
) -> List[MemoryFact]:
    """
    Semantic deduplication within a single distillation batch.

    Computes the full cosine similarity matrix in one vectorised NumPy
    operation, then greedily removes lower-scoring duplicates.

    Note: This deduplicates within a SINGLE distillation batch.
    Cross-run dedup against existing DB facts is handled by
    ``lt_store.upsert()`` (cosine >= 0.92 merge-on-insert).

    Args:
        facts: List of MemoryFact from one distillation run.
        similarity_threshold: Cosine threshold (0.85 default).
        embedder: Embedder with ``embed_documents(texts) -> List[List[float]]``.
                  If None, no dedup is performed (all facts kept).

    Returns:
        Deduplicated list of MemoryFact (highest-scoring fact kept per group).
    """
    if not facts or len(facts) <= 1:
        return facts

    if embedder is None:
        return facts

    try:
        texts = [f.text for f in facts]
        raw_embeddings = embedder.embed_documents(texts)
    except Exception:
        return facts

    import numpy as np

    # (N, 1536) matrix — one row per fact
    emb_matrix = np.array(raw_embeddings, dtype=np.float32)

    # L2-normalise each row so dot product = cosine similarity
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    emb_matrix = emb_matrix / norms

    # Full cosine similarity matrix in one matmul — O(N^2 * D) but vectorised
    sim_matrix = emb_matrix @ emb_matrix.T  # (N, N)

    # Build scores array for fast comparison
    scores = np.array([f.score for f in facts], dtype=np.float32)

    # Greedy dedup: for each duplicate pair above threshold, drop the lower-scoring one
    keep = np.ones(len(facts), dtype=bool)

    for i in range(len(facts)):
        if not keep[i]:
            continue
        # Find all j > i that are duplicates of i and still alive
        dupes = np.where(
            (sim_matrix[i, i + 1:] >= similarity_threshold) & keep[i + 1:]
        )[0] + (i + 1)

        for j in dupes:
            if scores[i] >= scores[j]:
                keep[j] = False
            else:
                keep[i] = False
                break

    return [f for f, k in zip(facts, keep) if k]
