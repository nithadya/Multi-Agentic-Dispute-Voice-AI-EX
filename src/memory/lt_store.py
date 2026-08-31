"""
Long-term memory store using Supabase PostgreSQL + pgvector.

Implements semantic storage and retrieval of memory facts.
"""

from loguru import logger
import uuid
from datetime import datetime
from typing import List, Iterable
from sqlalchemy import select, update, and_, text
from memory.schemas import MemoryFact, ReminderIntent
from memory.policies import apply_decay, should_prune
from infrastructure.db.sql_client import mem_facts_table, get_session
def _to_datetime(timestamp: float | None) -> datetime | None:
    """Convert Unix timestamp to datetime for PostgreSQL."""
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp)


class LongTermMemoryStore:
    """
    Long-term memory store (Supabase Postgres + pgvector).
    
    Stores distilled facts with metadata and embeddings.
    Uses pgvector for semantic search.
    """
    
    def __init__(self, embedder):
        self.embedder = embedder
    
    # Cosine similarity threshold for cross-run deduplication.
    # If a new fact is ≥ this similar to an existing DB fact, we
    # bump the existing fact's score instead of inserting a duplicate.
    DEDUP_SIMILARITY: float = 0.92

    def upsert(self, facts: Iterable[MemoryFact]) -> None:
        """Insert or update memory facts with semantic deduplication.

        For each incoming fact we first check if a semantically similar
        fact already exists in the DB (cosine ≥ DEDUP_SIMILARITY).
        If yes → bump the existing row's score and ``last_used_at``.
        If no  → insert the new fact.

        This prevents the "3× penicillin allergy" problem that occurs
        when overlapping ST turns are distilled across multiple runs.
        """
        facts_list = list(facts)
        if not facts_list:
            return

        session = get_session()
        try:
            texts = [fact.text for fact in facts_list]
            embeddings = self.embedder.embed_documents(texts)

            inserted = 0
            merged = 0

            for fact, embedding in zip(facts_list, embeddings):
                embedding_str = str(embedding)

                # ── Semantic dedup: check if a near-identical fact exists ──
                existing = session.execute(
                    text("""
                        SELECT id::text,
                               score,
                               1 - (embedding <=> CAST(:emb AS vector)) AS sim
                        FROM mem_facts
                        WHERE user_id = :uid
                          AND deleted = FALSE
                          AND 1 - (embedding <=> CAST(:emb AS vector)) >= :threshold
                        ORDER BY embedding <=> CAST(:emb AS vector)
                        LIMIT 1
                    """),
                    {
                        "emb": embedding_str,
                        "uid": fact.user_id,
                        "threshold": self.DEDUP_SIMILARITY,
                    },
                ).first()

                if existing:
                    # Bump score (take the higher) and refresh timestamp
                    new_score = max(existing.score, fact.score)
                    session.execute(
                        update(mem_facts_table)
                        .where(mem_facts_table.c.id == existing.id)
                        .values(
                            score=new_score,
                            last_used_at=_to_datetime(fact.last_used_at),
                        )
                    )
                    merged += 1
                    logger.debug(
                        "Merged with existing fact {} (sim={:.3f})",
                        existing.id,
                        existing.sim,
                    )
                else:
                    session.execute(
                        mem_facts_table.insert().values(
                            id=fact.id,
                            user_id=fact.user_id,
                            text=fact.text,
                            embedding=embedding,
                            score=fact.score,
                            tags=fact.tags,
                            created_at=_to_datetime(fact.created_at),
                            last_used_at=_to_datetime(fact.last_used_at),
                            ttl_at=_to_datetime(fact.ttl_at),
                            pin=fact.pin,
                            deleted=False,
                        )
                    )
                    inserted += 1
                    logger.debug("Inserted new fact: {}", fact.id)

            session.commit()
            logger.info(
                "Upserted {} facts to LT memory ({} new, {} merged)",
                len(facts_list),
                inserted,
                merged,
            )
        except Exception as e:
            session.rollback()
            logger.error("Failed to upsert facts: {}", e)
            raise
        finally:
            session.close()
    
    def query(self, user_id: str, query_text: str, k: int, threshold: float) -> List[MemoryFact]:
        """Query long-term memory with semantic search using pgvector."""
        query_embedding = self.embedder.embed_query(query_text)
        session = get_session()
        
        try:
            embedding_str = str(query_embedding)
            query_sql = text("""
                SELECT 
                    id::text,
                    user_id,
                    text,
                    score,
                    tags,
                    created_at,
                    last_used_at,
                    ttl_at,
                    pin,
                    1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
                FROM mem_facts
                WHERE user_id = :user_id
                    AND deleted = FALSE
                    AND (ttl_at IS NULL OR ttl_at > NOW())
                    AND 1 - (embedding <=> CAST(:embedding AS vector)) >= :threshold
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT :k
            """)
            
            # IVFFlat with few rows needs higher probes to avoid missing results
            session.execute(text("SET ivfflat.probes = 10"))
            results = session.execute(
                query_sql,
                {"embedding": embedding_str, "user_id": user_id, "threshold": threshold, "k": k}
            ).fetchall()
            
            facts = []
            for row in results:
                fact = MemoryFact(
                    id=row.id,
                    user_id=row.user_id,
                    text=row.text,
                    score=row.score,
                    tags=row.tags or [],
                    created_at=row.created_at.timestamp() if row.created_at else 0.0,
                    last_used_at=row.last_used_at.timestamp() if row.last_used_at else 0.0,
                    ttl_at=row.ttl_at.timestamp() if row.ttl_at else None,
                    pin=row.pin,
                )
                facts.append(fact)
            
            logger.info(f"Retrieved {len(facts)} facts from LT memory for user {user_id}")
            return facts
        except Exception as e:
            logger.error(f"Failed to query facts: {e}")
            raise
        finally:
            session.close()
    
    def get_all_facts(self, user_id: str) -> List[MemoryFact]:
        """Get all non-deleted facts for a user."""
        session = get_session()
        try:
            results = session.execute(
                select(mem_facts_table).where(
                    and_(
                        mem_facts_table.c.user_id == user_id,
                        mem_facts_table.c.deleted == False
                    )
                )
            ).fetchall()
            
            facts = []
            for row in results:
                fact = MemoryFact(
                    id=str(row.id),
                    user_id=row.user_id,
                    text=row.text,
                    score=row.score,
                    tags=row.tags or [],
                    created_at=row.created_at.timestamp() if row.created_at else 0.0,
                    last_used_at=row.last_used_at.timestamp() if row.last_used_at else 0.0,
                    ttl_at=row.ttl_at.timestamp() if row.ttl_at else None,
                    pin=row.pin,
                )
                facts.append(fact)
            return facts
        finally:
            session.close()
    
    def soft_delete(self, fact_id: str) -> None:
        """Soft-delete a fact (mark as deleted without removing)."""
        session = get_session()
        try:
            session.execute(
                update(mem_facts_table).where(
                    mem_facts_table.c.id == fact_id
                ).values(deleted=True)
            )
            session.commit()
            logger.info(f"Soft-deleted fact: {fact_id}")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to delete fact: {e}")
            raise
        finally:
            session.close()
    
    def prune(self, user_id: str) -> int:
        """Prune expired/low-value facts for a user."""
        session = get_session()
        try:
            facts = self.get_all_facts(user_id)
            pruned_count = 0
            for fact in facts:
                if should_prune(fact):
                    self.soft_delete(fact.id)
                    pruned_count += 1
            logger.info(f"Pruned {pruned_count} facts for user {user_id}")
            return pruned_count
        finally:
            session.close()
    
    def update_scores(self, user_id: str) -> None:
        """Apply temporal decay to all fact scores."""
        session = get_session()
        try:
            facts = self.get_all_facts(user_id)
            for fact in facts:
                new_score = apply_decay(fact)
                session.execute(
                    update(mem_facts_table).where(
                        mem_facts_table.c.id == fact.id
                    ).values(score=new_score)
                )
            session.commit()
            logger.info(f"Updated scores for {len(facts)} facts")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update scores: {e}")
            raise
        finally:
            session.close()
