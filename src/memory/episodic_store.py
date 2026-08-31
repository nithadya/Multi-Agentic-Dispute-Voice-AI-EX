"""
Episodic Memory Store - stores and retrieves full conversation episodes.

Migrated to Supabase (Postgres + pgvector).
"""

from loguru import logger
import json
import time
import uuid
from typing import List, Optional, Tuple
from sqlalchemy import text
from datetime import datetime

from memory.schemas import Episode, ConversationTurn
from infrastructure.db.supabase_client import get_supabase_session, set_user_context
class EpisodicMemoryStore:
    """
    Stores and retrieves complete conversation episodes (episodic long-term memory).
    """
    
    def __init__(self, embedder):
        self.embedder = embedder
        logger.info("✓ Episodic memory store initialized (Supabase/pgvector)")
    
    def store_episode(self, episode: Episode):
        """Store a conversation episode in long-term episodic memory."""
        session = get_supabase_session()
        try:
            set_user_context(episode.user_id)
            embeddings = self.embedder.embed_documents([episode.summary])
            embedding = embeddings[0]
            
            session.execute(
                text("""
                    INSERT INTO mem_episodes 
                    (id, user_id, session_id, summary, summary_embedding, topic_tags, 
                     start_at, end_at, turn_count, turns, created_at)
                    VALUES (:id, :user_id, :session_id, :summary, :summary_embedding, :topic_tags, 
                            :start_at, :end_at, :turn_count, :turns, :created_at)
                    ON CONFLICT (id) DO UPDATE SET
                        summary = EXCLUDED.summary,
                        summary_embedding = EXCLUDED.summary_embedding,
                        topic_tags = EXCLUDED.topic_tags,
                        turn_count = EXCLUDED.turn_count,
                        turns = EXCLUDED.turns
                """),
                {
                    "id": episode.id,
                    "user_id": episode.user_id,
                    "session_id": episode.session_id,
                    "summary": episode.summary,
                    "summary_embedding": embedding,
                    "topic_tags": json.dumps(episode.topic_tags),
                    "start_at": datetime.fromtimestamp(episode.start_at),
                    "end_at": datetime.fromtimestamp(episode.end_at),
                    "turn_count": episode.turn_count,
                    "turns": json.dumps([t.to_dict() for t in episode.turns]),
                    "created_at": datetime.now(),
                }
            )
            session.commit()
            logger.info(f"✓ Stored episode {episode.id} with {episode.turn_count} turns")
        except Exception as e:
            logger.error(f"Failed to store episode: {e}")
            session.rollback()
            raise
        finally:
            session.close()
    
    def query_episodes(
        self, user_id: str, query: str, k: int = 3,
        threshold: float = 0.5, time_range: Optional[Tuple[float, float]] = None
    ) -> List[Episode]:
        """Query episodic memory for relevant conversation episodes."""
        session = get_supabase_session()
        try:
            set_user_context(user_id)
            query_embedding = self.embedder.embed_query(query)
            
            if time_range:
                start_dt = datetime.fromtimestamp(time_range[0])
                end_dt = datetime.fromtimestamp(time_range[1])
                results = session.execute(
                    text("""
                        SELECT id, user_id, session_id, summary, topic_tags, start_at, end_at, 
                               turn_count, turns,
                               1 - (summary_embedding <=> CAST(:query_embedding AS vector)) AS similarity
                        FROM mem_episodes
                        WHERE user_id = :user_id
                          AND start_at >= :start_filter AND end_at <= :end_filter
                          AND 1 - (summary_embedding <=> CAST(:query_embedding AS vector)) >= :threshold
                        ORDER BY summary_embedding <=> CAST(:query_embedding AS vector)
                        LIMIT :k
                    """),
                    {
                        "query_embedding": str(query_embedding),
                        "user_id": user_id,
                        "start_filter": start_dt,
                        "end_filter": end_dt,
                        "threshold": threshold,
                        "k": k,
                    }
                ).fetchall()
            else:
                results = session.execute(
                    text("""
                        SELECT id, user_id, session_id, summary, topic_tags, start_at, end_at, 
                               turn_count, turns,
                               1 - (summary_embedding <=> CAST(:query_embedding AS vector)) AS similarity
                        FROM mem_episodes
                        WHERE user_id = :user_id
                          AND 1 - (summary_embedding <=> CAST(:query_embedding AS vector)) >= :threshold
                        ORDER BY summary_embedding <=> CAST(:query_embedding AS vector)
                        LIMIT :k
                    """),
                    {
                        "query_embedding": str(query_embedding),
                        "user_id": user_id,
                        "threshold": threshold,
                        "k": k,
                    }
                ).fetchall()
            
            episodes = []
            for result in results:
                turns_data = result.turns if isinstance(result.turns, list) else json.loads(result.turns)
                turns = [ConversationTurn.from_dict(t) for t in turns_data]
                
                episode = Episode(
                    id=result.id,
                    user_id=result.user_id,
                    session_id=result.session_id,
                    summary=result.summary,
                    topic_tags=result.topic_tags if isinstance(result.topic_tags, list) else (json.loads(result.topic_tags) if result.topic_tags else []),
                    start_at=result.start_at.timestamp(),
                    end_at=result.end_at.timestamp(),
                    turn_count=result.turn_count,
                    turns=turns
                )
                episodes.append(episode)
            
            logger.info(f"Retrieved {len(episodes)} episodes for user {user_id}")
            return episodes
        except Exception as e:
            logger.error(f"Failed to query episodes: {e}")
            raise
        finally:
            session.close()
    
    def get_episode_by_session(self, user_id: str, session_id: str) -> Optional[Episode]:
        """Retrieve a specific episode by session ID."""
        session = get_supabase_session()
        try:
            set_user_context(user_id)
            result = session.execute(
                text("""
                    SELECT id, user_id, session_id, summary, topic_tags, start_at, end_at, 
                           turn_count, turns
                    FROM mem_episodes 
                    WHERE user_id = :user_id AND session_id = :session_id
                    ORDER BY start_at DESC LIMIT 1
                """),
                {"user_id": user_id, "session_id": session_id}
            ).fetchone()
            
            if not result:
                return None
            
            turns_data = result.turns if isinstance(result.turns, list) else json.loads(result.turns)
            turns = [ConversationTurn.from_dict(t) for t in turns_data]
            
            return Episode(
                id=result.id, user_id=result.user_id, session_id=result.session_id,
                summary=result.summary,
                topic_tags=result.topic_tags if isinstance(result.topic_tags, list) else (json.loads(result.topic_tags) if result.topic_tags else []),
                start_at=result.start_at.timestamp(), end_at=result.end_at.timestamp(),
                turn_count=result.turn_count, turns=turns
            )
        except Exception as e:
            logger.error(f"Failed to get episode by session: {e}")
            raise
        finally:
            session.close()
    
    def list_recent_episodes(self, user_id: str, limit: int = 10, days_ago: int = 30) -> List[Episode]:
        """List recent episodes for a user."""
        session = get_supabase_session()
        try:
            set_user_context(user_id)
            cutoff_time = datetime.fromtimestamp(time.time() - (days_ago * 86400))
            
            results = session.execute(
                text("""
                    SELECT id, user_id, session_id, summary, topic_tags, start_at, end_at, 
                           turn_count, turns
                    FROM mem_episodes 
                    WHERE user_id = :user_id AND start_at >= :cutoff
                    ORDER BY start_at DESC LIMIT :limit
                """),
                {"user_id": user_id, "cutoff": cutoff_time, "limit": limit}
            ).fetchall()
            
            episodes = []
            for result in results:
                turns_data = result.turns if isinstance(result.turns, list) else json.loads(result.turns)
                turns = [ConversationTurn.from_dict(t) for t in turns_data]
                
                episode = Episode(
                    id=result.id, user_id=result.user_id, session_id=result.session_id,
                    summary=result.summary,
                    topic_tags=result.topic_tags if isinstance(result.topic_tags, list) else (json.loads(result.topic_tags) if result.topic_tags else []),
                    start_at=result.start_at.timestamp(), end_at=result.end_at.timestamp(),
                    turn_count=result.turn_count, turns=turns
                )
                episodes.append(episode)
            return episodes
        except Exception as e:
            logger.error(f"Failed to list recent episodes: {e}")
            raise
        finally:
            session.close()


def create_episode_from_turns(user_id: str, session_id: str, turns: List[ConversationTurn], llm=None) -> Episode:
    """Create an Episode from a list of conversation turns."""
    if not turns:
        raise ValueError("Cannot create episode from empty turns list")
    
    episode_id = str(uuid.uuid4())
    start_at = turns[0].ts
    end_at = turns[-1].ts
    turn_count = len(turns)
    
    if llm:
        conversation_text = "\n".join([f"{turn.role}: {turn.content}" for turn in turns])
        prompt = f"""Summarize this conversation in 1-2 sentences. Focus on the main topics discussed:\n\n{conversation_text}\n\nSummary:"""
        try:
            response = llm.invoke(prompt)
            summary = response.content if hasattr(response, 'content') else str(response)
            summary = summary.strip()
        except Exception as e:
            logger.warning(f"Failed to generate LLM summary: {e}")
            summary = f"Conversation with {turn_count} turns"
    else:
        summary = f"Conversation with {turn_count} turns from {session_id}"
    
    topic_tags = []
    keywords = ["medication", "allergy", "appointment", "doctor", "reminder", "pain", "symptom"]
    conversation_lower = " ".join([t.content.lower() for t in turns])
    for keyword in keywords:
        if keyword in conversation_lower:
            topic_tags.append(keyword)
    
    return Episode(
        id=episode_id, user_id=user_id, session_id=session_id,
        turns=turns, summary=summary, topic_tags=topic_tags,
        start_at=start_at, end_at=end_at, turn_count=turn_count
    )
