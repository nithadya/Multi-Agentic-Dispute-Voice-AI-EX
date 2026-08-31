"""
Initialize Neo4j Graph Schema for E-Commerce Dispute Resolution System.

Creates constraints and indexes for the vendor policy knowledge graph.

Run:
    cd src && python ../scripts/init_ecommerce_graph.py

Requires:
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in .env
"""

import os
import sys

# Add src to path
_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(_SRC))

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
from infrastructure.db.neo4j_client import get_neo4j_client

# ── Schema Definitions ─────────────────────────────────────────────────────

CONSTRAINTS = [
    # Uniqueness constraints (also create backing indexes)
    "CREATE CONSTRAINT vendor_policy_id IF NOT EXISTS FOR (p:VendorPolicy) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT dispute_type_name IF NOT EXISTS FOR (dt:DisputeType) REQUIRE dt.name IS UNIQUE",
    "CREATE CONSTRAINT product_category_name IF NOT EXISTS FOR (pc:ProductCategory) REQUIRE pc.name IS UNIQUE",
    "CREATE CONSTRAINT policy_condition_id IF NOT EXISTS FOR (c:PolicyCondition) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT resolution_id IF NOT EXISTS FOR (r:Resolution) REQUIRE r.id IS UNIQUE",
    "CREATE CONSTRAINT evidence_type IF NOT EXISTS FOR (e:Evidence) REQUIRE e.evidence_type IS UNIQUE",
    "CREATE CONSTRAINT escalation_name IF NOT EXISTS FOR (esc:EscalationPath) REQUIRE esc.path_name IS UNIQUE",
]

INDEXES = [
    # Lookup indexes for common query patterns
    "CREATE INDEX policy_category IF NOT EXISTS FOR (p:VendorPolicy) ON (p.applies_to_category)",
    "CREATE INDEX policy_max_days IF NOT EXISTS FOR (p:VendorPolicy) ON (p.max_return_days)",
    "CREATE INDEX condition_max_days IF NOT EXISTS FOR (c:PolicyCondition) ON (c.max_days)",
    "CREATE INDEX resolution_decision IF NOT EXISTS FOR (r:Resolution) ON (r.decision)",
]


def main():
    logger.info("Initializing Neo4j schema for E-Commerce Dispute Resolution...")

    client = get_neo4j_client()

    if not client.verify_connectivity():
        logger.error("Cannot connect to Neo4j. Check NEO4J_URI and NEO4J_PASSWORD in .env")
        sys.exit(1)

    # ── Apply Constraints ──────────────────────────────────────────────────
    logger.info("Creating constraints...")
    for cypher in CONSTRAINTS:
        try:
            client.write(cypher)
            logger.success("  ✓ {}", cypher.split("FOR")[0].strip().replace("CREATE CONSTRAINT ", ""))
        except Exception as exc:
            logger.warning("  ⚠ Constraint skipped (may already exist): {}", exc)

    # ── Apply Indexes ──────────────────────────────────────────────────────
    logger.info("Creating indexes...")
    for cypher in INDEXES:
        try:
            client.write(cypher)
            logger.success("  ✓ {}", cypher.split("FOR")[0].strip().replace("CREATE INDEX ", ""))
        except Exception as exc:
            logger.warning("  ⚠ Index skipped (may already exist): {}", exc)

    # ── Verify ─────────────────────────────────────────────────────────────
    schema = client.get_schema_info()
    logger.success(
        "Schema initialized. Graph state: {} nodes, {} relationships",
        schema["total_nodes"],
        schema["total_relationships"],
    )
    logger.info("Labels defined: {}", list(schema["labels"].keys()))

    client.close()
    logger.success("Done! Neo4j schema is ready.")


if __name__ == "__main__":
    main()
