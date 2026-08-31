"""
Graph RAG Service — E-Commerce Vendor Policy Knowledge Graph Retrieval.

Inspired by Week 11's Agentic GraphRAG pipeline (02_agentic_graph_rag.ipynb),
adapted for e-commerce dispute resolution domain.

The "RAG Bottleneck" problem: Standard vector RAG retrieves text by similarity
but cannot traverse conditional logic like:
    "If item is electronics AND purchase > 14 days, refund is VOID
     UNLESS item arrived damaged (requires photo evidence)"

Solution: Model vendor policies as a Neo4j knowledge graph where policy rules
are nodes/edges. Cypher multi-hop traversal handles the conditional logic
that flat vector search cannot.

Graph Schema:
    VendorPolicy ──[APPLIES_TO]──► ProductCategory
    VendorPolicy ──[HAS_CONDITION]──► PolicyCondition
    PolicyCondition ──[LEADS_TO]──► Resolution
    Resolution ──[REQUIRES]──► Evidence
    DisputeType ──[GOVERNED_BY]──► VendorPolicy
    ProductCategory ──[BELONGS_TO]──► Vendor
    Resolution ──[ESCALATES_TO]──► EscalationPath

CRAG Workflow (same pattern as Week 11):
    1. Entity Extraction  → LLM extracts dispute_type, category, vendor from query
    2. Graph Retrieval    → Cypher single-hop + multi-hop queries
    3. Grader             → Are retrieved policy nodes relevant?
    4. Rewriter           → If not relevant, reformulate and retry (max 2 loops)
    5. Generator          → Synthesize policy-grounded decision
"""

import time
from typing import Any, Dict, List, Optional

from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from infrastructure.config import (
    GRAPH_RAG_ENABLED,
    GRAPH_MAX_HOPS,
    GRAPH_TOP_K_NODES,
    GRAPH_CRAG_MAX_RETRIES,
)
from infrastructure.observability import observe, update_current_observation


# ── Prompt Templates (Week 11 style) ──────────────────────────────────

ENTITY_EXTRACTION_PROMPT = """\
Extract the key entities from this e-commerce dispute query for graph retrieval.

Return ONLY a JSON object with these fields:
{{
  "dispute_type": "<return|refund|damaged|missing|fraud|warranty|chargeback|other>",
  "product_category": "<electronics|clothing|furniture|food|books|sports|other>",
  "vendor_name": "<vendor name or null>",
  "time_since_purchase_days": <number or null>,
  "conditions": ["<any specific conditions mentioned>"]
}}

Examples:
- "I received a damaged laptop 3 days ago" → {{"dispute_type": "damaged", "product_category": "electronics", "vendor_name": null, "time_since_purchase_days": 3, "conditions": ["damaged_on_arrival"]}}
- "I want to return shoes I bought 45 days ago from Nike" → {{"dispute_type": "return", "product_category": "clothing", "vendor_name": "Nike", "time_since_purchase_days": 45, "conditions": []}}
"""

GRADER_PROMPT = """\
You are a relevance grader for an e-commerce dispute policy system.

Assess whether the retrieved policy context is relevant to the dispute query.

A policy is RELEVANT if it:
- Matches the dispute type (return/refund/damaged/missing/fraud)
- Applies to the product category mentioned
- Contains applicable conditions or resolutions

Grade as "yes" if relevant, "no" if not relevant.
Return ONLY "yes" or "no".\
"""

REWRITER_PROMPT = """\
You are a query rewriter for an e-commerce policy knowledge graph.

The initial query failed to retrieve relevant policy nodes. Reformulate it:
1. Use standard dispute terminology (return, refund, damaged, missing, warranty)
2. Expand product categories (e.g. "phone" → "electronics")
3. Remove specific dates/amounts — focus on the dispute type
4. Separate compound queries into the most important sub-query

Return ONLY the reformulated query, nothing else.\
"""

GENERATOR_PROMPT = """\
You are an autonomous E-Commerce Dispute Resolution AI powered by a vendor policy knowledge graph.

Use the provided graph context to make a policy-compliant decision on the dispute.

Rules:
- ONLY use information from the provided graph policy context
- Cite specific policy rules (e.g., "Policy P-001: Electronics returns within 14 days → full refund")
- State your decision clearly: APPROVE / PARTIAL_REFUND / REJECT / ESCALATE
- Explain the specific policy conditions that led to this decision
- If evidence is required, list exactly what evidence is needed
- If the policy context is insufficient, recommend escalation to a human agent
- Always include a reference to the governing policy node

Format:
Decision: <APPROVE|PARTIAL_REFUND|REJECT|ESCALATE>
Policy Reference: <policy_id>
Reasoning: <clear explanation citing graph context>
Refund Amount: <amount or "N/A" if not applicable>
Required Evidence: <list or "None required">\
"""


class GraphRetriever:
    """
    Retrieves policy information from Neo4j using Cypher queries.

    Runs both single-hop (direct policy lookup) and multi-hop
    (traverse conditions → resolutions → evidence) queries.
    """

    def __init__(self) -> None:
        from infrastructure.db.neo4j_client import get_neo4j_client
        self._client = get_neo4j_client()

    def retrieve(
        self,
        dispute_type: str,
        product_category: str,
        vendor_name: Optional[str] = None,
        vendor_id: Optional[str] = None,
        time_since_purchase_days: Optional[int] = None,
        conditions: Optional[List[str]] = None,
    ) -> str:
        """
        Run Cypher queries to fetch relevant policy nodes and their relationships.

        Returns formatted policy context string for the LLM generator.
        """
        results = []

        # ── Query 1: Direct policy lookup (single-hop) ──────────────
        # Match policies governing this dispute type + product category
        q1 = """
        MATCH (dt:DisputeType {name: $dispute_type})-[:GOVERNED_BY]->(p:VendorPolicy)
        OPTIONAL MATCH (p)-[:APPLIES_TO]->(pc:ProductCategory)
        OPTIONAL MATCH (v:Vendor)-[:HAS_POLICY]->(p)
        WHERE (pc IS NULL OR toLower(pc.name) CONTAINS toLower($product_category)
              OR toLower($product_category) CONTAINS toLower(pc.name))
          AND ($vendor_id IS NULL OR v.id = $vendor_id OR v IS NULL)
        RETURN p.id AS policy_id, p.name AS policy_name,
               p.description AS description,
               p.max_return_days AS max_return_days,
               p.refund_type AS refund_type,
               pc.name AS category
        LIMIT $limit
        """
        try:
            r1 = self._client.query(q1, {
                "dispute_type": dispute_type,
                "product_category": product_category,
                "vendor_id": vendor_id,
                "limit": GRAPH_TOP_K_NODES,
            })
            if r1:
                results.append("=== Direct Policy Matches ===")
                for row in r1:
                    results.append(
                        f"Policy: {row.get('policy_name', 'Unknown')} (ID: {row.get('policy_id', 'N/A')})\n"
                        f"  Category: {row.get('category', 'All')}\n"
                        f"  Description: {row.get('description', '')}\n"
                        f"  Max Return Days: {row.get('max_return_days', 'N/A')}\n"
                        f"  Refund Type: {row.get('refund_type', 'N/A')}"
                    )
        except Exception as exc:
            logger.warning("Graph Q1 failed: {}", exc)

        # ── Query 2: Multi-hop — conditions → resolutions (CORE INNOVATION) ──
        # This is what vector RAG CANNOT do: traverse conditional logic chains
        q2 = """
        MATCH (dt:DisputeType {name: $dispute_type})-[:GOVERNED_BY]->(p:VendorPolicy)
              -[:HAS_CONDITION]->(c:PolicyCondition)-[:LEADS_TO]->(r:Resolution)
        OPTIONAL MATCH (p)-[:APPLIES_TO]->(pc:ProductCategory)
        OPTIONAL MATCH (v:Vendor)-[:HAS_POLICY]->(p)
        WHERE ($vendor_id IS NULL OR v.id = $vendor_id OR v IS NULL)
          AND (pc IS NULL OR toLower(pc.name) CONTAINS toLower($product_category)
               OR toLower($product_category) CONTAINS toLower(pc.name))
        OPTIONAL MATCH (r)-[:REQUIRES]->(e:Evidence)
        OPTIONAL MATCH (r)-[:ESCALATES_TO]->(esc:EscalationPath)
        RETURN p.id AS policy_id,
               c.condition_text AS condition,
               c.applies_when AS applies_when,
               r.decision AS decision,
               r.refund_percentage AS refund_pct,
               r.description AS resolution_desc,
               collect(e.evidence_type) AS required_evidence,
               esc.path_name AS escalation_path
        LIMIT $limit
        """
        try:
            r2 = self._client.query(q2, {
                "dispute_type": dispute_type,
                "product_category": product_category,
                "vendor_id": vendor_id,
                "limit": GRAPH_TOP_K_NODES,
            })
            if r2:
                results.append("\n=== Policy Conditions → Resolutions (Multi-hop) ===")
                for row in r2:
                    evi = ", ".join(row.get("required_evidence") or []) or "None"
                    results.append(
                        f"Policy {row.get('policy_id', 'N/A')}:\n"
                        f"  Condition: {row.get('condition', '')}\n"
                        f"  Applies When: {row.get('applies_when', '')}\n"
                        f"  Decision: {row.get('decision', 'N/A')} "
                        f"({row.get('refund_pct', 'N/A')}% refund)\n"
                        f"  Resolution: {row.get('resolution_desc', '')}\n"
                        f"  Required Evidence: {evi}\n"
                        f"  Escalation Path: {row.get('escalation_path', 'N/A')}"
                    )
        except Exception as exc:
            logger.warning("Graph Q2 (multi-hop) failed: {}", exc)

        # ── Query 3: Time-based conditions (if purchase age known) ───
        if time_since_purchase_days is not None:
            q3 = """
            MATCH (p:VendorPolicy)-[:HAS_CONDITION]->(c:PolicyCondition)-[:LEADS_TO]->(r:Resolution)
            OPTIONAL MATCH (p)-[:APPLIES_TO]->(pc:ProductCategory)
            OPTIONAL MATCH (v:Vendor)-[:HAS_POLICY]->(p)
            WHERE c.max_days IS NOT NULL AND $days_since_purchase <= c.max_days
              AND ($vendor_id IS NULL OR v.id = $vendor_id OR v IS NULL)
              AND (pc IS NULL OR toLower(pc.name) CONTAINS toLower($product_category)
                   OR toLower($product_category) CONTAINS toLower(pc.name))
            OPTIONAL MATCH (dt:DisputeType)-[:GOVERNED_BY]->(p)
            WHERE dt.name = $dispute_type OR dt.name IS NULL
            RETURN p.id AS policy_id, c.condition_text AS condition,
                   c.max_days AS max_days, r.decision AS decision,
                   r.refund_percentage AS refund_pct
            LIMIT $limit
            """
            try:
                r3 = self._client.query(q3, {
                    "days_since_purchase": time_since_purchase_days,
                    "dispute_type": dispute_type,
                    "product_category": product_category,
                    "vendor_id": vendor_id,
                    "limit": 5,
                })
                if r3:
                    results.append(f"\n=== Time-Based Policy Rules (Purchase age: {time_since_purchase_days} days) ===")
                    for row in r3:
                        results.append(
                            f"Policy {row.get('policy_id', 'N/A')}: "
                            f"{row.get('condition', '')} "
                            f"(within {row.get('max_days', 'N/A')} days) → "
                            f"{row.get('decision', 'N/A')} ({row.get('refund_pct', 'N/A')}%)"
                        )
            except Exception as exc:
                logger.warning("Graph Q3 (time-based) failed: {}", exc)

        if not results:
            return ""

        return "\n".join(results)


class GraphRAGService:
    """
    Agentic Graph RAG with CRAG-style self-correction loop.

    Workflow (mirrors Week 11's notebook 02):
        Router    → dispute_type / product_category entity extraction
        Retriever → Cypher single-hop + multi-hop
        Grader    → is retrieved context relevant?
        Rewriter  → if not relevant, rewrite query and retry (max N times)
        Generator → synthesize policy-compliant decision
    """

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.retriever = GraphRetriever()

    @observe(name="graph_rag_query")
    def query(
        self,
        user_query: str,
        verbose: bool = False,
        vendor_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the full Graph RAG pipeline for a dispute query.

        Returns:
            answer: Policy-grounded decision text
            policy_context: Raw graph context used
            retrieval_attempts: How many CRAG retries were needed
            entities: Extracted entities dict
            graph_used: True (always, since this is graph_rag_service)
        """
        start = time.time()

        # ── Step 1: Entity Extraction ────────────────────────────────
        entities = self._extract_entities(user_query)
        if verbose:
            logger.info("Graph entities: {}", entities)

        dispute_type = entities.get("dispute_type", "return")
        product_category = entities.get("product_category", "other")
        vendor_name = entities.get("vendor_name")
        time_days = entities.get("time_since_purchase_days")
        conditions = entities.get("conditions", [])

        # ── Step 2–4: CRAG loop (retrieve → grade → rewrite) ─────────
        current_query = user_query
        policy_context = ""
        retrieval_attempts = 0

        for attempt in range(GRAPH_CRAG_MAX_RETRIES + 1):
            retrieval_attempts = attempt + 1

            # Retrieve
            policy_context = self.retriever.retrieve(
                dispute_type=dispute_type,
                product_category=product_category,
                vendor_name=vendor_name,
                vendor_id=vendor_id,
                time_since_purchase_days=time_days,
                conditions=conditions,
            )

            if not policy_context:
                if verbose:
                    logger.warning("Graph retrieval returned empty. Attempt {}", attempt + 1)
                # Try broadening the dispute type before rewriting
                if dispute_type != "other" and attempt == 0:
                    dispute_type = "other"
                    continue
                if attempt < GRAPH_CRAG_MAX_RETRIES:
                    current_query = self._rewrite_query(current_query)
                    dispute_type, product_category = self._extract_simple_entities(current_query)
                continue

            # Grade
            is_relevant = self._grade(policy_context, current_query)
            if verbose:
                logger.info("Graph grader: relevant={}", is_relevant)

            if is_relevant:
                break

            # Rewrite for next attempt
            if attempt < GRAPH_CRAG_MAX_RETRIES:
                current_query = self._rewrite_query(current_query)
                dispute_type, product_category = self._extract_simple_entities(current_query)

        # ── Step 5: Generate answer ───────────────────────────────────
        if policy_context:
            answer = self._generate(policy_context, user_query)
        else:
            answer = (
                "No specific vendor policy found in the knowledge graph for this dispute. "
                "Escalating to human review for manual policy interpretation."
            )

        elapsed_ms = int((time.time() - start) * 1000)
        update_current_observation(
            metadata={
                "graph_entities": entities,
                "retrieval_attempts": retrieval_attempts,
                "graph_context_length": len(policy_context),
                "latency_ms": elapsed_ms,
            }
        )

        return {
            "answer": answer,
            "policy_context": policy_context,
            "retrieval_attempts": retrieval_attempts,
            "entities": entities,
            "graph_used": True,
            "latency_ms": elapsed_ms,
        }

    # ── Internal helpers ───────────────────────────────────────────────

    def _extract_entities(self, query: str) -> Dict:
        """Use LLM to extract dispute entities from the user query."""
        import json

        prompt = ChatPromptTemplate.from_messages([
            ("system", ENTITY_EXTRACTION_PROMPT),
            ("human", "Query: {query}"),
        ])
        chain = prompt | self.llm | StrOutputParser()

        try:
            raw = chain.invoke({"query": query})
            # Strip markdown fences
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start:end + 1])
        except Exception as exc:
            logger.warning("Entity extraction failed: {}", exc)

        return {
            "dispute_type": "return",
            "product_category": "other",
            "vendor_name": None,
            "time_since_purchase_days": None,
            "conditions": [],
        }

    def _extract_simple_entities(self, query: str):
        """Quick keyword-based entity extraction for rewrite loop fallback."""
        dispute_keywords = {
            "return": ["return", "send back", "give back"],
            "refund": ["refund", "money back", "reimbursement"],
            "damaged": ["damaged", "broken", "defective", "faulty"],
            "missing": ["missing", "not arrived", "lost", "didn't receive"],
            "fraud": ["fraud", "scam", "unauthorized", "stolen"],
        }
        category_keywords = {
            "electronics": ["phone", "laptop", "computer", "tablet", "electronic", "tech"],
            "clothing": ["shirt", "shoes", "dress", "clothing", "apparel", "fashion"],
            "furniture": ["furniture", "chair", "table", "sofa", "desk"],
            "food": ["food", "groceries", "meal", "drink"],
        }

        q_lower = query.lower()
        dispute_type = "return"
        for dtype, keywords in dispute_keywords.items():
            if any(kw in q_lower for kw in keywords):
                dispute_type = dtype
                break

        product_category = "other"
        for cat, keywords in category_keywords.items():
            if any(kw in q_lower for kw in keywords):
                product_category = cat
                break

        return dispute_type, product_category

    def _grade(self, context: str, query: str) -> bool:
        """Grade whether the graph context is relevant to the query."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", GRADER_PROMPT),
            ("human", "Context:\n{context}\n\nQuery: {query}"),
        ])
        chain = prompt | self.llm | StrOutputParser()
        try:
            result = chain.invoke({"context": context[:2000], "query": query})
            return "yes" in result.lower()
        except Exception:
            return True  # Optimistic default — don't block on grader failure

    def _rewrite_query(self, query: str) -> str:
        """Rewrite the query for better graph entity extraction."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", REWRITER_PROMPT),
            ("human", "Original query: {query}\n\nRewrite for better policy retrieval:"),
        ])
        chain = prompt | self.llm | StrOutputParser()
        try:
            return chain.invoke({"query": query}).strip()
        except Exception:
            return query  # Return original if rewrite fails

    def _generate(self, policy_context: str, query: str) -> str:
        """Generate a policy-grounded decision from the graph context."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", GENERATOR_PROMPT),
            ("human", "Graph Policy Context:\n{context}\n\nDispute Query: {query}"),
        ])
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": policy_context, "query": query})


__all__ = ["GraphRAGService", "GraphRetriever"]
