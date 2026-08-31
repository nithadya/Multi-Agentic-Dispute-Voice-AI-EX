"""
Qdrant ingestion pipeline — load, chunk, embed, upsert.

This module contains the core service logic for ingesting documents
into Qdrant Cloud.  Scripts (``scripts/ingest_to_qdrant.py``) and CLI
commands should call :func:`run_ingest` rather than duplicating the
pipeline steps.
"""

import json
from loguru import logger
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from infrastructure.config import (
    MARKDOWN_DIR,
    JSONL_DIR,
    KB_DIR,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_BATCH_SIZE,
)
from infrastructure.llm import get_default_embeddings
from infrastructure.db.qdrant_client import (
    ensure_collection,
    delete_collection,
    upsert_chunks,
    collection_info,
)
from services.ingest_service import (
    semantic_chunk,
    fixed_chunk,
    sliding_chunk,
    parent_child_chunk,
)
# =====================================================================
# Strategy registry
# =====================================================================

STRATEGY_MAP = {
    "semantic": semantic_chunk,
    "fixed": fixed_chunk,
    "sliding": sliding_chunk,
    "parent_child": parent_child_chunk,
}


# =====================================================================
# Document loaders
# =====================================================================


def load_kb_docs(kb_dir: Path | None = None) -> List[Dict[str, Any]]:
    """Load internal knowledge-base markdown documents."""
    kb_dir = Path(kb_dir or KB_DIR)
    if not kb_dir.exists():
        raise FileNotFoundError(f"Knowledge-base directory not found: {kb_dir}")

    docs: List[Dict[str, Any]] = []
    for md_file in sorted(kb_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        title = content.split("\n", 1)[0].lstrip("# ").strip() or md_file.stem
        doc_slug = md_file.stem.lstrip("0123456789_")
        url = f"internal://disputeai/{doc_slug}"
        docs.append({"url": url, "title": title, "content": content})

    logger.info("Loaded {} knowledge-base documents from {}", len(docs), kb_dir)
    return docs


def load_markdown_docs(md_dir: Path | None = None) -> List[Dict[str, Any]]:
    """Load crawled markdown files from disk."""
    md_dir = Path(md_dir or MARKDOWN_DIR)
    if not md_dir.exists():
        raise FileNotFoundError(f"Markdown directory not found: {md_dir}")

    docs: List[Dict[str, Any]] = []
    for md_file in sorted(md_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        title = content.split("\n", 1)[0].lstrip("# ").strip() or md_file.stem
        url = f"https://disputeai.com/{md_file.stem}"
        docs.append({"url": url, "title": title, "content": content})

    logger.info("Loaded {} markdown documents from {}", len(docs), md_dir)
    return docs


def load_jsonl_docs(jsonl_dir: Path | None = None) -> List[Dict[str, Any]]:
    """Load documents from JSONL crawl output."""
    jsonl_dir = Path(jsonl_dir or JSONL_DIR)
    if not jsonl_dir.exists():
        raise FileNotFoundError(f"JSONL directory not found: {jsonl_dir}")

    docs: List[Dict[str, Any]] = []
    for jsonl_file in sorted(jsonl_dir.glob("*.jsonl")):
        with open(jsonl_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("content"):
                    docs.append(
                        {
                            "url": obj.get("url", ""),
                            "title": obj.get("title", ""),
                            "content": obj["content"],
                        }
                    )

    logger.info("Loaded {} documents from JSONL in {}", len(docs), jsonl_dir)
    return docs


LOADER_MAP = {
    "kb": load_kb_docs,
    "markdown": load_markdown_docs,
    "jsonl": load_jsonl_docs,
}


# =====================================================================
# Embedding helper
# =====================================================================


def embed_texts(
    texts: List[str],
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> List[List[float]]:
    """Embed a list of texts using the configured embedding model."""
    embedder = get_default_embeddings(batch_size=batch_size)
    all_embeddings: List[List[float]] = []

    total_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(
            "Embedding batch {}/{} ({} texts)...",
            batch_num,
            total_batches,
            len(batch),
        )
        batch_embeddings = embedder.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


# =====================================================================
# Parent-child helpers
# =====================================================================


def _build_parent_lookup(parents: List[Dict[str, Any]]) -> Dict[str, str]:
    """Build a mapping from parent_id → parent text."""
    return {p["parent_id"]: p["text"] for p in parents}


def _enrich_children_with_parent_text(
    children: List[Dict[str, Any]],
    parent_lookup: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Attach ``parent_text`` to each child chunk for richer LLM context."""
    for child in children:
        pid = child.get("parent_id", "")
        child["parent_text"] = parent_lookup.get(pid, child["text"])
    return children


# =====================================================================
# Core pipeline
# =====================================================================


def run_ingest(
    source: str = "kb",
    strategy: str = "parent_child",
    recreate: bool = False,
) -> int:
    """
    End-to-end ingestion pipeline.

    Args:
        source: One of ``kb``, ``markdown``, ``jsonl``.
        strategy: One of ``semantic``, ``fixed``, ``sliding``, ``parent_child``.
        recreate: If ``True``, drop and recreate the Qdrant collection first.

    Returns:
        Number of points upserted.

    Raises:
        ValueError: If *source* or *strategy* is unknown.
        FileNotFoundError: If the source directory does not exist.
    """
    logger.info("=" * 70)
    logger.info("🚀 QDRANT INGESTION PIPELINE")
    logger.info("=" * 70)

    # ── 1. Load documents ────────────────────────────────────
    loader = LOADER_MAP.get(source)
    if loader is None:
        raise ValueError(
            f"Unknown source: {source}. Choose from {list(LOADER_MAP.keys())}"
        )

    logger.info(f"\n📂 Loading documents (source={source})...")
    docs = loader()
    if not docs:
        logger.error("❌ No documents loaded. Nothing to ingest.")
        sys.exit(1)

    # ── 2. Chunk ─────────────────────────────────────────────
    logger.info(f"\n✂️  Chunking (strategy={strategy})...")
    chunk_fn = STRATEGY_MAP.get(strategy)
    if chunk_fn is None:
        raise ValueError(
            f"Unknown strategy: {strategy}. Choose from {list(STRATEGY_MAP.keys())}"
        )

    if strategy == "parent_child":
        children, parents = chunk_fn(docs)
        logger.info(f"   → {len(children)} child chunks, {len(parents)} parent chunks")
        parent_lookup = _build_parent_lookup(parents)
        chunks = _enrich_children_with_parent_text(children, parent_lookup)
        logger.info("   → Each child enriched with parent_text for richer LLM context")
    else:
        chunks = chunk_fn(docs)
        logger.info(f"   → {len(chunks)} chunks created")

    if not chunks:
        logger.error("❌ No chunks produced. Check your documents.")
        sys.exit(1)

    # ── 3. Embed ─────────────────────────────────────────────
    logger.info(f"\n🔢 Embedding {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    t0 = time.time()
    embeddings = embed_texts(texts)
    embed_secs = time.time() - t0
    logger.success(f"   → Embedding done in {embed_secs:.1f}s")

    # ── 4. Create / recreate collection ──────────────────────
    if recreate:
        logger.info(f"\n🗑️  Recreating collection '{QDRANT_COLLECTION_NAME}'...")
        try:
            delete_collection()
        except Exception:
            pass  # collection may not exist yet

    ensure_collection()

    # ── 5. Upsert ────────────────────────────────────────────
    logger.info(f"\n⬆️  Upserting {len(chunks)} points into Qdrant...")
    t0 = time.time()
    n = upsert_chunks(chunks, embeddings)
    upsert_secs = time.time() - t0
    logger.info(f"   → Upserted {n} points in {upsert_secs:.1f}s")

    # ── 6. Verify ────────────────────────────────────────────
    logger.info("\n📊 Collection info:")
    info = collection_info()
    for k, v in info.items():
        logger.info(f"   {k}: {v}")

    logger.info("\n" + "=" * 70)
    logger.success("✅ INGESTION COMPLETE")
    logger.info(f"   Source: {source}")
    logger.info(f"   Strategy: {strategy}")
    logger.info(f"   Chunks indexed: {n}")
    if strategy == "parent_child":
        logger.info("   Parent context: Stored in payload for richer LLM generation")
    logger.info("=" * 70)

    return n
