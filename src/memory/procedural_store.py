"""
Procedural Memory Store - stores and retrieves workflows and procedures.

Uses pgvector for semantic retrieval.
"""

from loguru import logger
from typing import Any, Dict, List, Optional
from datetime import datetime
from uuid import UUID
import json

from sqlalchemy import text
from infrastructure.db.sql_client import get_sql_engine
from infrastructure.llm import get_default_embeddings
from memory.schemas import Procedure
class ProceduralMemoryStore:
    """Store and retrieve procedural knowledge (workflows, step-by-step guides)."""
    
    def __init__(self):
        self.engine = get_sql_engine()
        self.embeddings = get_default_embeddings()
        logger.debug("ProceduralMemoryStore initialized with Supabase pgvector")
    
    def store_procedure(
        self, name: str, description: str, steps: List[Dict[str, Any]],
        context_when: Optional[str] = None, conditions: Optional[Dict[str, Any]] = None,
        examples: Optional[List[str]] = None, category: Optional[str] = None,
    ) -> UUID:
        """Store a new procedure in the database."""
        embed_text = f"{description}. Context: {context_when or 'General'}"
        embedding = self.embeddings.embed_query(embed_text)
        embedding_str = str(embedding)
        
        query = text("""
            INSERT INTO mem_procedures (
                name, description, context_when, steps, conditions, 
                examples, embedding, category
            )
            VALUES (
                :name, :description, :context_when, 
                CAST(:steps AS jsonb), CAST(:conditions AS jsonb),
                CAST(:examples AS jsonb), CAST(:embedding AS vector), :category
            )
            RETURNING id
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(
                query,
                {
                    "name": name, "description": description,
                    "context_when": context_when, "steps": json.dumps(steps),
                    "conditions": json.dumps(conditions) if conditions else None,
                    "examples": json.dumps(examples) if examples else None,
                    "embedding": embedding_str, "category": category,
                }
            )
            conn.commit()
            procedure_id = result.fetchone()[0]
            logger.info(f"Stored procedure '{name}' with ID {procedure_id}")
            return procedure_id if isinstance(procedure_id, UUID) else UUID(procedure_id)
    
    def query_procedures(
        self, query_text: str, top_k: int = 3,
        threshold: float = 0.3, category: Optional[str] = None,
    ) -> List[Procedure]:
        """Semantically search for relevant procedures based on query text."""
        query_embedding = self.embeddings.embed_query(query_text)
        query_embedding_str = str(query_embedding)
        
        if category:
            query = text("""
                SELECT id, name, description, steps, context_when, conditions, examples, category,
                       1 - (embedding <=> CAST(:query_embedding AS vector)) AS similarity
                FROM mem_procedures
                WHERE active = TRUE AND category = :category
                    AND 1 - (embedding <=> CAST(:query_embedding AS vector)) >= :threshold
                ORDER BY embedding <=> CAST(:query_embedding AS vector) LIMIT :top_k
            """)
            params = {"query_embedding": query_embedding_str, "threshold": threshold, "top_k": top_k, "category": category}
        else:
            query = text("""
                SELECT id, name, description, steps, context_when, conditions, examples, category,
                       1 - (embedding <=> CAST(:query_embedding AS vector)) AS similarity
                FROM mem_procedures
                WHERE active = TRUE
                    AND 1 - (embedding <=> CAST(:query_embedding AS vector)) >= :threshold
                ORDER BY embedding <=> CAST(:query_embedding AS vector) LIMIT :top_k
            """)
            params = {"query_embedding": query_embedding_str, "threshold": threshold, "top_k": top_k}
        
        with self.engine.connect() as conn:
            # IVFFlat with few rows needs higher probes to avoid missing results
            conn.execute(text("SET ivfflat.probes = 10"))
            results = conn.execute(query, params).fetchall()
        
        procedures = []
        for row in results:
            steps = row.steps if isinstance(row.steps, list) else []
            conditions = row.conditions if isinstance(row.conditions, (dict, type(None))) else None
            examples = row.examples if isinstance(row.examples, list) else []
            
            procedures.append(Procedure(
                id=str(row.id), name=row.name, description=row.description,
                steps=steps, context_when=row.context_when, conditions=conditions,
                examples=examples, category=row.category, similarity=row.similarity,
            ))
        
        logger.debug(f"Found {len(procedures)} procedures for query: '{query_text[:50]}...'")
        return procedures
    
    def get_procedure_by_name(self, name: str) -> Optional[Procedure]:
        """Get a specific procedure by its unique name."""
        query = text("""
            SELECT id, name, description, steps, context_when, conditions, examples, category
            FROM mem_procedures WHERE name = :name AND active = TRUE
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {"name": name}).fetchone()
        
        if not result:
            return None
        
        steps = result.steps if isinstance(result.steps, list) else []
        conditions = result.conditions if isinstance(result.conditions, (dict, type(None))) else None
        examples = result.examples if isinstance(result.examples, list) else []
        
        return Procedure(
            id=str(result.id), name=result.name, description=result.description,
            steps=steps, context_when=result.context_when, conditions=conditions,
            examples=examples, category=result.category,
        )
    
    def list_all_procedures(self, category: Optional[str] = None) -> List[Procedure]:
        """List all active procedures, optionally filtered by category."""
        with self.engine.connect() as conn:
            if category:
                query = text("""
                    SELECT id, name, description, steps, context_when, conditions, examples, category
                    FROM mem_procedures WHERE active = TRUE AND category = :category ORDER BY name
                """)
                results = conn.execute(query, {"category": category}).fetchall()
            else:
                query = text("""
                    SELECT id, name, description, steps, context_when, conditions, examples, category
                    FROM mem_procedures WHERE active = TRUE ORDER BY category, name
                """)
                results = conn.execute(query).fetchall()
        
        procedures = []
        for row in results:
            steps = row.steps if isinstance(row.steps, list) else []
            conditions = row.conditions if isinstance(row.conditions, (dict, type(None))) else None
            examples = row.examples if isinstance(row.examples, list) else []
            
            procedures.append(Procedure(
                id=str(row.id), name=row.name, description=row.description,
                steps=steps, context_when=row.context_when, conditions=conditions,
                examples=examples, category=row.category,
            ))
        return procedures
    
    def update_procedure(
        self, name: str, description: Optional[str] = None,
        steps: Optional[List[Dict[str, Any]]] = None, context_when: Optional[str] = None,
        conditions: Optional[Dict[str, Any]] = None, examples: Optional[List[str]] = None,
        category: Optional[str] = None,
    ) -> bool:
        """Update an existing procedure (creates a new version)."""
        current = self.get_procedure_by_name(name)
        if not current:
            logger.warning(f"Procedure '{name}' not found for update")
            return False
        
        new_desc = description or current.description
        new_steps = steps or current.steps
        new_context = context_when or current.context_when
        new_conditions = conditions or current.conditions
        new_examples = examples or current.examples
        new_category = category or current.category
        
        embed_text = f"{new_desc}. Context: {new_context or 'General'}"
        embedding = self.embeddings.embed_query(embed_text)
        embedding_str = str(embedding)
        
        query = text("""
            UPDATE mem_procedures
            SET description = :description, steps = CAST(:steps AS jsonb),
                context_when = :context_when, conditions = CAST(:conditions AS jsonb),
                examples = CAST(:examples AS jsonb), category = :category,
                embedding = CAST(:embedding AS vector), version = version + 1, updated_at = NOW()
            WHERE name = :name AND active = TRUE
        """)
        
        with self.engine.connect() as conn:
            conn.execute(query, {
                "name": name, "description": new_desc, "steps": json.dumps(new_steps),
                "context_when": new_context,
                "conditions": json.dumps(new_conditions) if new_conditions else None,
                "examples": json.dumps(new_examples) if new_examples else None,
                "category": new_category, "embedding": embedding_str,
            })
            conn.commit()
        
        logger.info(f"Updated procedure '{name}' to new version")
        return True
    
    def deactivate_procedure(self, name: str) -> bool:
        """Soft-delete a procedure by marking it inactive."""
        query = text("""
            UPDATE mem_procedures SET active = FALSE, updated_at = NOW()
            WHERE name = :name AND active = TRUE
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {"name": name})
            conn.commit()
            if result.rowcount > 0:
                logger.info(f"Deactivated procedure '{name}'")
                return True
            else:
                logger.warning(f"Procedure '{name}' not found or already inactive")
                return False
