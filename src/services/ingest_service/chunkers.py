"""
Text chunking strategies for document ingestion.

Provides 5 chunking strategies:
1. Semantic/Heading-Aware - Split by document structure
2. Fixed-Window - Uniform chunks with overlap
3. Sliding-Window - Overlapping windows for better recall
4. Parent-Child (Two-Tier) - Small children with large parent context
5. Query-Focused Late Chunking - Large base passages, split on retrieval

All strategies use configuration from infrastructure.config
"""

from typing import List, Dict, Any, Optional, Tuple
import tiktoken
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)

from infrastructure.config import (
    FIXED_CHUNK_SIZE,
    FIXED_CHUNK_OVERLAP,
    SEMANTIC_MAX_CHUNK_SIZE,
    SEMANTIC_MIN_CHUNK_SIZE,
    SLIDING_WINDOW_SIZE,
    SLIDING_STRIDE_SIZE,
    PARENT_CHUNK_SIZE,
    CHILD_CHUNK_SIZE,
    CHILD_OVERLAP,
    LATE_CHUNK_BASE_SIZE,
    LATE_CHUNK_SPLIT_SIZE,
    LATE_CHUNK_CONTEXT_WINDOW
)
from infrastructure.models import Document, Chunk


# ============================================================================
# Utility Functions
# ============================================================================

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


# ============================================================================
# 1. SEMANTIC / HEADING-AWARE CHUNKING
# ============================================================================

def semantic_chunk(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split documents by markdown heading structure.
    
    Use when: Documents have clear heading hierarchy
    Pros: Preserves topic coherence
    Cons: Variable chunk sizes
    
    Args:
        documents: List of dicts with 'url', 'title', 'content'
    
    Returns:
        List of chunk dicts with 'url', 'title', 'text', 'strategy', 'chunk_index'
    """
    chunks = []
    chunk_idx = 0
    
    # Define heading hierarchy
    headers_to_split = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split,
        strip_headers=False
    )
    
    for doc in documents:
        content = doc['content']
        url = doc['url']
        title = doc['title']
        
        try:
            # Split by headings
            sections = splitter.split_text(content)
            
            if not sections:
                # No headings, use full content
                sections = [type('obj', (object,), {'page_content': content, 'metadata': {}})()]
            
            for section in sections:
                text = section.page_content.strip()
                
                if not text or len(text) < SEMANTIC_MIN_CHUNK_SIZE:
                    continue
                
                # If section too large, split further
                if count_tokens(text) > SEMANTIC_MAX_CHUNK_SIZE:
                    # Recursive split
                    char_size = SEMANTIC_MAX_CHUNK_SIZE * 4
                    sub_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=char_size,
                        chunk_overlap=100,
                        length_function=len
                    )
                    sub_chunks = sub_splitter.split_text(text)
                    
                    for sub_text in sub_chunks:
                        if sub_text.strip():
                            chunks.append({
                                "url": url,
                                "title": title,
                                "text": sub_text.strip(),
                                "strategy": "semantic",
                                "chunk_index": chunk_idx,
                                "heading": section.metadata.get('h1', '') or section.metadata.get('h2', '')
                            })
                            chunk_idx += 1
                else:
                    chunks.append({
                        "url": url,
                        "title": title,
                        "text": text,
                        "strategy": "semantic",
                        "chunk_index": chunk_idx,
                        "heading": section.metadata.get('h1', '') or section.metadata.get('h2', '')
                    })
                    chunk_idx += 1
                    
        except Exception as e:
            # Fallback: treat as single chunk
            if content.strip():
                chunks.append({
                    "url": url,
                    "title": title,
                    "text": content.strip(),
                    "strategy": "semantic",
                    "chunk_index": chunk_idx,
                    "heading": ""
                })
                chunk_idx += 1
    
    return chunks


# ============================================================================
# 2. FIXED-WINDOW CHUNKING
# ============================================================================

def fixed_chunk(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split documents into fixed-size chunks with overlap.
    
    Use when: Need predictable chunk sizes for embedding
    Pros: Uniform sizes, simple
    Cons: Breaks semantic boundaries
    
    Args:
        documents: List of dicts with 'url', 'title', 'content'
    
    Returns:
        List of chunk dicts with 'url', 'title', 'text', 'strategy', 'chunk_index'
    """
    chunks = []
    chunk_idx = 0
    
    # Character approximation: ~4 chars per token
    chunk_size_chars = FIXED_CHUNK_SIZE * 4
    overlap_chars = FIXED_CHUNK_OVERLAP * 4
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_chars,
        chunk_overlap=overlap_chars,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    for doc in documents:
        content = doc['content']
        url = doc['url']
        title = doc['title']
        
        # Split content
        doc_chunks = splitter.split_text(content)
        
        for text in doc_chunks:
            if text.strip():
                token_count = count_tokens(text)
                chunks.append({
                    "url": url,
                    "title": title,
                    "text": text.strip(),
                    "strategy": "fixed",
                    "chunk_index": chunk_idx,
                    "token_count": token_count,
                    "overlap_tokens": FIXED_CHUNK_OVERLAP
                })
                chunk_idx += 1
    
    return chunks


# ============================================================================
# 3. SLIDING-WINDOW CHUNKING
# ============================================================================

def sliding_chunk(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create overlapping sliding windows for better recall.
    
    Use when: Need better coverage of content
    Pros: Better recall, no missed boundaries
    Cons: More chunks (index bloat)
    
    Args:
        documents: List of dicts with 'url', 'title', 'content'
    
    Returns:
        List of chunk dicts with 'url', 'title', 'text', 'strategy', 'chunk_index'
    """
    chunks = []
    chunk_idx = 0
    
    # Window parameters from config
    window_size_chars = SLIDING_WINDOW_SIZE * 4
    stride_chars = SLIDING_STRIDE_SIZE * 4
    
    for doc in documents:
        content = doc['content']
        url = doc['url']
        title = doc['title']
        
        # Simple sliding window over content
        pos = 0
        window_idx = 0
        content_len = len(content)
        
        while pos < content_len:
            end = min(pos + window_size_chars, content_len)
            window_text = content[pos:end]
            
            if window_text.strip():
                chunks.append({
                    "url": url,
                    "title": title,
                    "text": window_text.strip(),
                    "strategy": "sliding",
                    "chunk_index": chunk_idx,
                    "window_index": window_idx,
                    "overlap_tokens": SLIDING_STRIDE_SIZE if window_idx > 0 else 0
                })
                chunk_idx += 1
                window_idx += 1
            
            # Move by stride
            pos += stride_chars
            if pos >= content_len:
                break
    
    return chunks


# ============================================================================
# 4. PARENT-CHILD (TWO-TIER) CHUNKING
# ============================================================================

def parent_child_chunk(documents: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Create parent-child chunk pairs for precise retrieval with rich context.
    
    How it works:
    1. Split document into large "parent" chunks (1200 tokens)
    2. Within each parent, create small "child" chunks (250 tokens)
    3. Store children in index with parent_id reference
    4. On retrieval: fetch children, return parent context to LLM
    
    Use when: Want precise retrieval but rich context for generation
    Pros: Best of both worlds - precision + context
    Cons: More complex retrieval logic needed
    
    Returns:
        Tuple of (children_chunks, parent_chunks)
        Children have 'parent_id' field linking to parent
    """
    parent_chunks = []
    child_chunks = []
    parent_idx = 0
    child_idx = 0
    
    # Character approximations
    parent_size_chars = PARENT_CHUNK_SIZE * 4
    child_size_chars = CHILD_CHUNK_SIZE * 4
    child_overlap_chars = CHILD_OVERLAP * 4
    
    # Parent splitter
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size_chars,
        chunk_overlap=200,  # Small overlap between parents
        length_function=len
    )
    
    # Child splitter
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size_chars,
        chunk_overlap=child_overlap_chars,
        length_function=len
    )
    
    for doc in documents:
        content = doc['content']
        url = doc['url']
        title = doc['title']
        
        # Create parent chunks
        parent_texts = parent_splitter.split_text(content)
        
        for parent_text in parent_texts:
            if not parent_text.strip():
                continue
            
            parent_id = f"{url}::parent::{parent_idx}"
            
            # Store parent
            parent_chunks.append({
                "parent_id": parent_id,
                "url": url,
                "title": title,
                "text": parent_text.strip(),
                "strategy": "parent",
                "chunk_index": parent_idx,
                "token_count": count_tokens(parent_text)
            })
            
            # Create children within this parent
            child_texts = child_splitter.split_text(parent_text)
            
            for child_text in child_texts:
                if child_text.strip():
                    child_chunks.append({
                        "child_id": f"{parent_id}::child::{child_idx}",
                        "parent_id": parent_id,
                        "url": url,
                        "title": title,
                        "text": child_text.strip(),
                        "strategy": "child",
                        "chunk_index": child_idx,
                        "token_count": count_tokens(child_text)
                    })
                    child_idx += 1
            
            parent_idx += 1
    
    return child_chunks, parent_chunks


# ============================================================================
# 5. QUERY-FOCUSED LATE CHUNKING
# ============================================================================

def late_chunk_index(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create large base passages for indexing (split on retrieval).
    
    How it works:
    1. Index large passages (1000 tokens)
    2. On retrieval, split near query matches into smaller chunks
    3. Provides tighter quotes without exploding index size
    
    Use when: Need precision without pre-micro-chunking everything
    Pros: Smaller index, better match density
    Cons: Requires custom retrieval logic
    
    Returns:
        List of base passage chunks (to be split later on query)
    """
    chunks = []
    chunk_idx = 0
    
    # Large base passages
    base_size_chars = LATE_CHUNK_BASE_SIZE * 4
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=base_size_chars,
        chunk_overlap=100,
        length_function=len
    )
    
    for doc in documents:
        content = doc['content']
        url = doc['url']
        title = doc['title']
        
        # Create base passages
        passages = splitter.split_text(content)
        
        for passage in passages:
            if passage.strip():
                chunks.append({
                    "url": url,
                    "title": title,
                    "text": passage.strip(),
                    "strategy": "late_chunk_base",
                    "chunk_index": chunk_idx,
                    "token_count": count_tokens(passage),
                    "splittable": True  # Mark for late splitting
                })
                chunk_idx += 1
    
    return chunks


def late_chunk_split(passage: str, query: str) -> List[Dict[str, Any]]:
    """
    Split a base passage near query matches for precise retrieval.
    
    This is called at RETRIEVAL TIME, not indexing time.
    
    Args:
        passage: The base passage text
        query: User query
    
    Returns:
        List of smaller chunks around query matches
    """
    # Find query term positions
    query_terms = query.lower().split()
    passage_lower = passage.lower()
    
    # Find all match positions
    match_positions = []
    for term in query_terms:
        pos = 0
        while True:
            pos = passage_lower.find(term, pos)
            if pos == -1:
                break
            match_positions.append(pos)
            pos += len(term)
    
    if not match_positions:
        # No matches, return full passage as one chunk
        return [{"text": passage, "score": 0.0}]
    
    # Create chunks around matches
    chunks = []
    context_chars = LATE_CHUNK_CONTEXT_WINDOW * 4
    split_size_chars = LATE_CHUNK_SPLIT_SIZE * 4
    
    for match_pos in match_positions:
        # Extract context around match
        start = max(0, match_pos - context_chars)
        end = min(len(passage), match_pos + split_size_chars)
        
        chunk_text = passage[start:end].strip()
        
        # Calculate relevance score (proximity to query)
        score = 1.0 if match_pos in match_positions else 0.5
        
        chunks.append({
            "text": chunk_text,
            "match_position": match_pos,
            "score": score
        })
    
    # Deduplicate overlapping chunks
    unique_chunks = []
    seen_texts = set()
    for chunk in sorted(chunks, key=lambda x: x['score'], reverse=True):
        if chunk['text'] not in seen_texts:
            unique_chunks.append(chunk)
            seen_texts.add(chunk['text'])
    
    return unique_chunks[:5]  # Return top 5 relevant splits


# ============================================================================
# Chunking Service Class
# ============================================================================

class ChunkingService:
    """
    Unified service for all chunking strategies.
    
    Usage:
        service = ChunkingService()
        chunks = service.chunk(documents, strategy="semantic")
    """
    
    def __init__(self):
        self.strategies = {
            "semantic": semantic_chunk,
            "fixed": fixed_chunk,
            "sliding": sliding_chunk,
            "parent_child": parent_child_chunk,
            "late_chunk": late_chunk_index
        }
    
    def chunk(
        self,
        documents: List[Dict[str, Any]],
        strategy: str = "semantic"
    ) -> Any:
        """
        Chunk documents using specified strategy.
        
        Args:
            documents: List of document dicts
            strategy: One of 'semantic', 'fixed', 'sliding', 'parent_child', 'late_chunk'
        
        Returns:
            List of chunks (or tuple for parent_child)
        """
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(self.strategies.keys())}")
        
        return self.strategies[strategy](documents)
    
    def available_strategies(self) -> List[str]:
        """Return list of available chunking strategies."""
        return list(self.strategies.keys())


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "semantic_chunk",
    "fixed_chunk",
    "sliding_chunk",
    "parent_child_chunk",
    "late_chunk_index",
    "late_chunk_split",
    "ChunkingService",
    "count_tokens"
]

