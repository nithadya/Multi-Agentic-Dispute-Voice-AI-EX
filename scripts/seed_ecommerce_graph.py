"""
Seed Neo4j with E-Commerce Vendor Policy Knowledge Graph.

Reads data/ecommerce_policies.json and creates the full knowledge graph:
    VendorPolicy ──[APPLIES_TO]──► ProductCategory
    VendorPolicy ──[HAS_CONDITION]──► PolicyCondition
    PolicyCondition ──[LEADS_TO]──► Resolution
    Resolution ──[REQUIRES]──► Evidence
    DisputeType ──[GOVERNED_BY]──► VendorPolicy

Run:
    cd src && python ../scripts/seed_ecommerce_graph.py

Prerequisites:
    Run init_ecommerce_graph.py first to create schema constraints.
"""

import json
import os
import sys
import uuid

# Add src to path
_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "ecommerce_policies.json")
sys.path.insert(0, os.path.abspath(_SRC))

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
from infrastructure.db.neo4j_client import get_neo4j_client


def load_seed_data():
    with open(_DATA, "r", encoding="utf-8") as f:
        return json.load(f)


def seed_vendor_policies(client, data):
    """Seed VendorPolicy nodes."""
    policies = data["vendor_policies"]
    logger.info("Seeding {} VendorPolicy nodes...", len(policies))

    cypher = """
    UNWIND $batch AS row
    MERGE (p:VendorPolicy {id: row.id})
    SET p.name = row.name,
        p.description = row.description,
        p.max_return_days = row.max_return_days,
        p.refund_type = row.refund_type
    """
    n = client.batch_write(cypher, policies)
    logger.success("  ✓ {} VendorPolicy nodes created/merged", n)


def seed_dispute_types(client, data):
    """Seed DisputeType nodes."""
    types = data["dispute_types"]
    logger.info("Seeding {} DisputeType nodes...", len(types))

    cypher = """
    UNWIND $batch AS row
    MERGE (dt:DisputeType {name: row.name})
    SET dt.description = row.description
    """
    n = client.batch_write(cypher, types)
    logger.success("  ✓ {} DisputeType nodes created/merged", n)


def seed_product_categories(client, data):
    """Seed ProductCategory nodes."""
    categories = data["product_categories"]
    logger.info("Seeding {} ProductCategory nodes...", len(categories))

    cypher = """
    UNWIND $batch AS row
    MERGE (pc:ProductCategory {name: row.name})
    SET pc.description = row.description
    """
    n = client.batch_write(cypher, categories)
    logger.success("  ✓ {} ProductCategory nodes created/merged", n)


def seed_policy_relationships(client, data):
    """Create VendorPolicy -[APPLIES_TO]-> ProductCategory relationships."""
    logger.info("Creating APPLIES_TO relationships...")

    count = 0
    for policy in data["vendor_policies"]:
        for cat_name in policy.get("applies_to_categories", []):
            client.write("""
            MATCH (p:VendorPolicy {id: $policy_id})
            MATCH (pc:ProductCategory {name: $cat_name})
            MERGE (p)-[:APPLIES_TO]->(pc)
            """, {"policy_id": policy["id"], "cat_name": cat_name})
            count += 1

    logger.success("  ✓ {} APPLIES_TO relationships created", count)


def seed_dispute_type_policy_relationships(client, data):
    """
    Create DisputeType -[GOVERNED_BY]-> VendorPolicy relationships.

    Business rules:
    - 'damaged'  → P-005 (Damaged Goods Policy) + all category policies
    - 'missing'  → P-006 (Missing Order Policy)
    - 'fraud'    → P-007 (Fraud Policy)
    - 'warranty' → P-008 (Warranty Policy)
    - 'return'   → P-001 (Standard) + P-002 (Electronics) + P-003 (Clothing) + P-004 (Furniture)
    - 'refund'   → P-001 (Standard)
    - 'chargeback' → P-007 (Fraud/Chargeback)
    """
    mappings = [
        ("damaged",    ["P-005"]),
        ("missing",    ["P-006"]),
        ("fraud",      ["P-007"]),
        ("warranty",   ["P-008"]),
        ("return",     ["P-001", "P-002", "P-003", "P-004", "P-009"]),
        ("refund",     ["P-001", "P-002", "P-003"]),
        ("chargeback", ["P-007"]),
        ("other",      ["P-001"]),
    ]

    logger.info("Creating GOVERNED_BY relationships...")
    count = 0
    for dispute_type, policy_ids in mappings:
        for policy_id in policy_ids:
            client.write("""
            MATCH (dt:DisputeType {name: $dispute_type})
            MATCH (p:VendorPolicy {id: $policy_id})
            MERGE (dt)-[:GOVERNED_BY]->(p)
            """, {"dispute_type": dispute_type, "policy_id": policy_id})
            count += 1

    logger.success("  ✓ {} GOVERNED_BY relationships created", count)


def seed_policy_conditions(client, data):
    """Seed PolicyCondition nodes and HAS_CONDITION + LEADS_TO + REQUIRES relationships."""
    conditions = data["policy_conditions"]
    logger.info("Seeding {} PolicyCondition nodes + resolutions...", len(conditions))

    count = 0
    for cond in conditions:
        cond_id = f"COND-{str(uuid.uuid4())[:8].upper()}"
        res_id = f"RES-{str(uuid.uuid4())[:8].upper()}"
        policy_id = cond["policy_id"]

        # Create PolicyCondition node
        client.write("""
        MERGE (c:PolicyCondition {id: $cond_id})
        SET c.condition_text = $condition_text,
            c.applies_when = $applies_when,
            c.max_days = $max_days
        """, {
            "cond_id": cond_id,
            "condition_text": cond["condition_text"],
            "applies_when": cond["applies_when"],
            "max_days": cond["max_days"],
        })

        # Create Resolution node
        client.write("""
        MERGE (r:Resolution {id: $res_id})
        SET r.decision = $decision,
            r.refund_percentage = $refund_percentage,
            r.description = $description
        """, {
            "res_id": res_id,
            "decision": cond["decision"],
            "refund_percentage": cond["refund_percentage"],
            "description": f"{cond['decision']} ({cond['refund_percentage']}% refund)",
        })

        # VendorPolicy -[HAS_CONDITION]-> PolicyCondition
        client.write("""
        MATCH (p:VendorPolicy {id: $policy_id})
        MATCH (c:PolicyCondition {id: $cond_id})
        MERGE (p)-[:HAS_CONDITION]->(c)
        """, {"policy_id": policy_id, "cond_id": cond_id})

        # PolicyCondition -[LEADS_TO]-> Resolution
        client.write("""
        MATCH (c:PolicyCondition {id: $cond_id})
        MATCH (r:Resolution {id: $res_id})
        MERGE (c)-[:LEADS_TO]->(r)
        """, {"cond_id": cond_id, "res_id": res_id})

        # Resolution -[REQUIRES]-> Evidence (for each required evidence type)
        for ev_type in cond.get("required_evidence", []):
            client.write("""
            MERGE (e:Evidence {evidence_type: $ev_type})
            WITH e
            MATCH (r:Resolution {id: $res_id})
            MERGE (r)-[:REQUIRES]->(e)
            """, {"ev_type": ev_type, "res_id": res_id})

        # ESCALATE resolutions get an EscalationPath
        if cond["decision"] == "ESCALATE":
            client.write("""
            MERGE (esc:EscalationPath {path_name: 'human_review_queue'})
            SET esc.description = 'Route to human dispute resolution specialist',
                esc.sla_hours = 24
            WITH esc
            MATCH (r:Resolution {id: $res_id})
            MERGE (r)-[:ESCALATES_TO]->(esc)
            """, {"res_id": res_id})

        count += 1

    logger.success("  ✓ {} conditions + resolutions seeded", count)


def verify_graph(client):
    """Print final graph statistics."""
    schema = client.get_schema_info()
    logger.success("\n📊 Graph Statistics:")
    logger.info("  Total Nodes: {}", schema["total_nodes"])
    logger.info("  Total Relationships: {}", schema["total_relationships"])
    for label, count in schema["labels"].items():
        logger.info("  {} {}: {}", "📦", label, count)
    for rel_type, count in schema["relationship_types"].items():
        logger.info("  🔗 {}: {}", rel_type, count)


def main():
    logger.info("🚀 Seeding E-Commerce Policy Knowledge Graph into Neo4j...")

    client = get_neo4j_client()
    if not client.verify_connectivity():
        logger.error("Cannot connect to Neo4j. Run init_ecommerce_graph.py first.")
        sys.exit(1)

    data = load_seed_data()

    seed_vendor_policies(client, data)
    seed_dispute_types(client, data)
    seed_product_categories(client, data)
    seed_policy_relationships(client, data)
    seed_dispute_type_policy_relationships(client, data)
    seed_policy_conditions(client, data)

    verify_graph(client)

    client.close()
    logger.success("\n✅ Knowledge graph seeded successfully! Ready for Graph RAG queries.")
    logger.info("Test: python -c \"from infrastructure.db.neo4j_client import get_neo4j_client; print(get_neo4j_client().get_schema_info())\"")


if __name__ == "__main__":
    main()
