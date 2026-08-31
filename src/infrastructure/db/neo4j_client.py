"""
Neo4j Aura client — context-manager interface for the e-commerce policy knowledge graph.

Ported from Week 11 (utils/neo4j_client.py) and extended for the dispute resolution domain.
All queries use parameterized Cypher (never string interpolation for values).
Supports batch writes via UNWIND for efficient bulk ingestion.
"""

from typing import Any, Dict, List, Optional
from loguru import logger


class Neo4jClient:
    """Thread-safe Neo4j driver wrapper with context-manager support."""

    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        from neo4j import GraphDatabase

        self._uri = uri
        self._database = database
        self._driver = GraphDatabase.driver(uri, auth=(username, password))

    # -- Context Manager --

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # -- Connection --

    def verify_connectivity(self) -> bool:
        """Test that the driver can reach the server."""
        try:
            self._driver.verify_connectivity()
            logger.success(f"Connected to Neo4j at {self._uri}")
            return True
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            return False

    def close(self):
        """Release the driver and all pooled connections."""
        self._driver.close()

    # -- Read --

    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a read query and return list of record dicts."""
        params = params or {}
        with self._driver.session(database=self._database) as session:
            result = session.run(cypher, **params)
            return [dict(record) for record in result]

    # -- Write --

    def write(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Execute a single write statement."""
        params = params or {}
        with self._driver.session(database=self._database) as session:
            session.run(cypher, **params)

    def batch_write(self, cypher: str, param_list: List[Dict[str, Any]]) -> int:
        """Execute a write for each param dict in the list using UNWIND.

        The cypher should reference `$batch` as an UNWIND variable.
        Example:
            UNWIND $batch AS row
            MERGE (p:VendorPolicy {id: row.id})
            SET p.name = row.name

        Returns the number of items processed.
        """
        if not param_list:
            return 0
        with self._driver.session(database=self._database) as session:
            session.run(cypher, batch=param_list)
        return len(param_list)

    # -- Introspection --

    def get_node_count(self, label: Optional[str] = None) -> int:
        """Count nodes, optionally filtered by label."""
        if label:
            result = self.query(f"MATCH (n:`{label}`) RETURN count(n) AS cnt")
        else:
            result = self.query("MATCH (n) RETURN count(n) AS cnt")
        return result[0]["cnt"] if result else 0

    def get_relationship_count(self, rel_type: Optional[str] = None) -> int:
        """Count relationships, optionally filtered by type."""
        if rel_type:
            result = self.query(f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) AS cnt")
        else:
            result = self.query("MATCH ()-[r]->() RETURN count(r) AS cnt")
        return result[0]["cnt"] if result else 0

    def get_schema_info(self) -> Dict[str, Any]:
        """Introspect the graph for labels, relationship types, and counts."""
        labels = self.query("CALL db.labels() YIELD label RETURN label")
        rel_types = self.query(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        )

        info = {
            "labels": {},
            "relationship_types": {},
            "total_nodes": self.get_node_count(),
            "total_relationships": self.get_relationship_count(),
        }

        for row in labels:
            lbl = row["label"]
            info["labels"][lbl] = self.get_node_count(lbl)

        for row in rel_types:
            rt = row["relationshipType"]
            info["relationship_types"][rt] = self.get_relationship_count(rt)

        return info


_neo4j_client: Optional[Neo4jClient] = None


def get_neo4j_client() -> Neo4jClient:
    """Singleton factory: create a Neo4jClient from environment config."""
    global _neo4j_client
    if _neo4j_client is not None:
        return _neo4j_client

    from infrastructure.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE

    if not NEO4J_URI:
        raise ValueError("NEO4J_URI is not set. Add it to your .env file.")
    if not NEO4J_PASSWORD:
        raise ValueError("NEO4J_PASSWORD is not set. Add it to your .env file.")

    _neo4j_client = Neo4jClient(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
    )
    return _neo4j_client


def neo4j_available() -> bool:
    """Returns True if NEO4J_URI is configured (doesn't actually connect)."""
    from infrastructure.config import NEO4J_URI, NEO4J_PASSWORD
    return bool(NEO4J_URI and NEO4J_PASSWORD)
