import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Ensure src in PYTHONPATH
sys.path.insert(0, os.path.abspath('src'))

from loguru import logger
from infrastructure.llm import get_chat_llm, get_default_embeddings
from infrastructure.config import load_faqs
from agents.tools.rag_tool import RAGTool

def rebuild_cag_cache():
    logger.info("Initializing Embedder and RAG Tool...")
    # NOTE: Since we hardcoded the answers, we don't strictly need LLM for generation,
    # but RAGTool expects it to initialize the CRAGService.
    # LLM is heavily used natively during Agent usage anyway.
    llm = get_chat_llm()
    embedder = get_default_embeddings()
    
    rag_tool = RAGTool(embedder=embedder, llm=llm)
    
    logger.info("1. Clearing the existing CAG cache...")
    rag_tool.clear_cache()
    
    logger.info("2. Loading hardcoded FAQs from config/faqs.yaml...")
    faqs = load_faqs()
    logger.info(f"   Loaded {len(faqs)} explicit query/answer pairs.")
    
    if not faqs:
        logger.error("No FAQs loaded. Make sure config/faqs.yaml exists and is formatted correctly.")
        return

    logger.info("3. Warming the CAG cache up...")
    cached_count = rag_tool.warm_cache(faqs)
    
    logger.success(f"Successfully cached {cached_count} items into the cag_cache.")
    
    logger.info("4. Testing the cache with a known query...")
    # Run a test query exactly mirroring the hardcoded one
    test_query = faqs[0]['query']
    result = rag_tool.search(test_query)
    logger.info(f"Test Query: {test_query}")
    logger.info(f"Output Answer from Cache: {result}")

if __name__ == "__main__":
    rebuild_cag_cache()
