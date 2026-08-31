"""
Core domain models.

Defines Document, Chunk, Evidence, and related data structures.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class Document:
    """
    Represents a crawled web document.

    Attributes:
        url: Source URL of the document
        title: Document or page title
        content: Full text content (Markdown)
        metadata: Additional metadata (headings, links, depth, etc.)
    """
    url: str
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.url:
            raise ValueError("Document URL cannot be empty")
        if not self.content:
            raise ValueError("Document content cannot be empty")


@dataclass
class Chunk:
    """
    Represents a text chunk from a document.

    Attributes:
        text: The chunk content
        strategy: Chunking strategy used (semantic/fixed/sliding)
        chunk_index: Position in the original document
        url: Source document URL
        title: Source document title
        metadata: Additional metadata
    """
    text: str
    strategy: str
    chunk_index: int
    url: str
    title: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.strategy not in ["semantic", "fixed", "sliding"]:
            raise ValueError(f"Invalid strategy: {self.strategy}")


@dataclass
class Evidence:
    """
    Represents retrieved evidence for RAG.

    Attributes:
        url: Source URL
        title: Source title
        quote: Text excerpt (first ~400 chars)
        strategy: Chunking strategy
        score: Similarity/relevance score (optional)
        metadata: Additional metadata
    """
    url: str
    title: str
    quote: str
    strategy: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGQuery:
    """
    Represents a RAG query with context.

    Attributes:
        query: User question
        k: Number of documents to retrieve
        confidence_threshold: Minimum confidence for CRAG
        use_cache: Whether to check CAG cache
    """
    query: str
    k: int = 4
    confidence_threshold: float = 0.6
    use_cache: bool = True


@dataclass
class RAGResponse:
    """
    Represents a RAG response with metadata.

    Attributes:
        answer: Generated answer text
        evidence: List of Evidence objects
        confidence: Confidence score (for CRAG)
        cache_hit: Whether answer came from cache (for CAG)
        generation_time: Time taken to generate (seconds)
        metadata: Additional metadata (correction applied, etc.)
    """
    answer: str
    evidence: list[Evidence]
    confidence: Optional[float] = None
    cache_hit: bool = False
    generation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
