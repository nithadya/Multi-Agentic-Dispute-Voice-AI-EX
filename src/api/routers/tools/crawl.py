"""
Web crawler tool — REST wrapper around ``DisputeAIWebCrawler``.

The crawler is already async (Playwright via ``crawl_async``) so this
endpoint awaits it directly. Crawls can take seconds-to-minutes; the
client should use a generous HTTP read timeout.
"""

import time

from fastapi import APIRouter

from api.schemas import CrawledDoc, CrawlRequest, CrawlResponse


router = APIRouter(prefix="/tools/crawl", tags=["Tools — Crawler"])


@router.post("", response_model=CrawlResponse)
async def crawl(req: CrawlRequest) -> CrawlResponse:
    from services.ingest_service.web_crawler import DisputeAIWebCrawler

    t0 = time.perf_counter()
    crawler = DisputeAIWebCrawler(
        base_url=req.base_url,
        max_depth=req.max_depth,
        exclude_patterns=req.exclude_patterns or [],
    )
    raw = await crawler.crawl_async(req.start_urls, request_delay=req.request_delay)

    docs = [
        CrawledDoc(
            url=d.get("url", ""),
            title=d.get("title", ""),
            headings=d.get("headings", []) or [],
            content=d.get("content", ""),
            depth_level=int(d.get("depth_level", 0) or 0),
        )
        for d in (raw or [])
    ]
    return CrawlResponse(
        doc_count=len(docs),
        docs=docs,
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )
