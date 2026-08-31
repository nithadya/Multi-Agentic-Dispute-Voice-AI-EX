"""
Tests for memory policies - scoring, decay, pruning.
"""

import pytest
import time
from memory.policies import (
    score_memory_fact,
    apply_decay,
    should_prune,
)
from memory.schemas import MemoryFact


def test_score_memory_fact():
    """Test memory fact scoring."""
    now = time.time()
    created_at = now - 86400  # 1 day ago
    
    # Recent fact with explicit keyword
    score = score_memory_fact(
        text="Remember to always take meds at 8am",
        created_at=created_at,
        now=now,
        repetition_count=3,
    )
    
    assert 0.0 <= score <= 1.0
    assert score > 0.5  # Should be high due to keywords


def test_apply_decay():
    """Test decay application."""
    fact = MemoryFact(
        id="test1",
        user_id="user1",
        text="Test fact",
        score=0.8,
        created_at=time.time() - 86400 * 30,  # 30 days ago
        last_used_at=time.time() - 86400 * 30,
    )
    
    now = time.time()
    decayed_score = apply_decay(fact, now, half_life_days=30)
    
    assert decayed_score < fact.score  # Should decay
    assert decayed_score > 0.0  # But not to zero


def test_should_prune():
    """Test pruning decision."""
    now = time.time()
    
    # Low score fact - should prune
    fact_low = MemoryFact(
        id="test1",
        user_id="user1",
        text="Test",
        score=0.05,
        created_at=now - 86400,
        last_used_at=now - 86400,
    )
    
    assert should_prune(fact_low, now, ttl_seconds=86400 * 90, min_score=0.1)
    
    # Pinned fact - never prune
    fact_pinned = MemoryFact(
        id="test2",
        user_id="user1",
        text="Test",
        score=0.05,
        created_at=now - 86400 * 100,
        last_used_at=now - 86400 * 100,
        pin=True,
    )
    
    assert not should_prune(fact_pinned, now, ttl_seconds=86400 * 90)



