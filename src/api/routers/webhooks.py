from fastapi import APIRouter, HTTPException, Request
from loguru import logger
import json

from api.schemas import WebhookPayload, ErrorResponse
from infrastructure.db.neo4j_client import get_neo4j_client
from infrastructure.db.qdrant_client import get_qdrant_client
from infrastructure.llm.embeddings import get_default_embeddings
from infrastructure.config import QDRANT_COLLECTION_NAME

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])

def _format_policy_to_markdown(record) -> str:
    """Converts the structured JSON policy into natural language Markdown for vector search."""
    lines = [
        f"# Policy: {record.policy_name or record.policy_type.capitalize()}",
        f"**Vendor ID**: {record.vendor_id}",
        f"**Policy Type**: {record.policy_type}",
    ]
    if record.max_return_days is not None:
        lines.append(f"- **Maximum Return Window**: {record.max_return_days} days.")
    if record.refund_type:
        lines.append(f"- **Refund Type**: {record.refund_type}")
    if record.restocking_fee_percent is not None:
        lines.append(f"- **Restocking Fee**: {record.restocking_fee_percent}%")
    
    if record.conditions:
        lines.append("\n**Additional Conditions:**")
        for key, value in record.conditions.items():
            clean_key = key.replace("_", " ").capitalize()
            lines.append(f"- {clean_key}: {value}")
            
    return "\n".join(lines)


@router.post(
    "/policy-sync",
    summary="Sync Vendor Policy from Supabase",
    responses={
        200: {"description": "Synced successfully"},
        400: {"model": ErrorResponse},
    },
)
async def sync_vendor_policy(payload: WebhookPayload, request: Request):
    """
    Webhook receiver for Supabase.
    Triggered when a vendor policy is INSERTED or UPDATED.
    """
    record = payload.record
    
    # We only care about approved policies
    if not record.approved_by_admin:
        logger.info("Ignoring webhook for unapproved policy {} (Vendor: {})", record.id, record.vendor_id)
        return {"status": "ignored", "reason": "not approved"}

    try:
        # 1. Sync to Neo4j (Graph)
        neo4j_client = get_neo4j_client()
        cypher = """
        MERGE (v:Vendor {id: $vendor_id})
        MERGE (p:VendorPolicy {id: $policy_id})
        SET p.type = $type,
            p.name = $name,
            p.max_return_days = $max_return_days,
            p.refund_type = $refund_type,
            p.restocking_fee_percent = $restocking_fee_percent,
            p.conditions = $conditions
        MERGE (v)-[:HAS_POLICY]->(p)
        """
        params = {
            "vendor_id": record.vendor_id,
            "policy_id": record.id,
            "type": record.policy_type,
            "name": record.policy_name,
            "max_return_days": record.max_return_days,
            "refund_type": record.refund_type,
            "restocking_fee_percent": float(record.restocking_fee_percent) if record.restocking_fee_percent is not None else 0.0,
            "conditions": json.dumps(record.conditions or {})
        }
        neo4j_client.execute_write(cypher, params)
        logger.info("Synced policy {} to Neo4j", record.id)

        # 2. Sync to Qdrant (Vector)
        qdrant_client = get_qdrant_client()
        embedder = get_default_embeddings()
        
        md_text = _format_policy_to_markdown(record)
        vector = embedder.embed_query(md_text)
        
        payload_dict = {
            "chunk_id": record.id,
            "parent_id": f"vendor_{record.vendor_id}",
            "vendor_id": record.vendor_id,
            "title": record.policy_name or record.policy_type,
            "chunk_text": md_text,
            "type": "vendor_policy"
        }
        
        # We use a hash of the ID as the integer ID required by Qdrant
        point_id = abs(hash(record.id)) % (10 ** 15)
        
        from qdrant_client.models import PointStruct
        point = PointStruct(
            id=point_id,
            vector=vector,
            payload=payload_dict
        )
        
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[point]
        )
        logger.info("Synced policy {} to Qdrant", record.id)

        return {"status": "success", "policy_id": record.id}

    except Exception as e:
        logger.error("Failed to sync policy: {}", str(e))
        raise HTTPException(status_code=500, detail=str(e))
