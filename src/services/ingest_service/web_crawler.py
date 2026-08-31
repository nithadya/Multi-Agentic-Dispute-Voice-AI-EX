"""
Web crawler service for extracting content from websites.

NOTE: This module is NOT used in the current pipeline. The knowledge base
documents (data/knowledge_base/*.md) were created using this crawler once
and are now committed as static files. The ingestion pipeline reads those
local markdown files directly via pipeline.py.

Keep this module for future re-crawling if the website content changes.

Provides:
- DisputeAIWebCrawler: Playwright-based async crawler for JS-rendered sites
- Content extraction with BeautifulSoup
- Markdown conversion with markdownify
- Polite crawling with depth control and rate limiting
"""

from loguru import logger
from typing import List, Dict, Any, Set
from collections import deque
import re
import asyncio
from urllib.parse import urljoin, urlparse

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from markdownify import markdownify as md


class DisputeAIWebCrawler:
    """
    Async web crawler using Playwright for JavaScript-rendered content.
    
    Features:
    - Respects depth limits and exclude patterns
    - Handles SPA/React apps with proper JS rendering waits
    - Extracts clean markdown content
    - Discovers internal links for BFS traversal
    - Polite crawling with configurable delays
    
    Usage:
        crawler = DisputeAIWebCrawler(
            base_url="https://www.disputeai.com",
            max_depth=3,
            exclude_patterns=["/login", "/admin"]
        )
        documents = crawler.crawl(start_urls)
    """
    
    def __init__(self, base_url: str, max_depth: int, exclude_patterns: List[str]):
        self.base_url = base_url
        self.max_depth = max_depth
        self.exclude_patterns = exclude_patterns
        self.visited: Set[str] = set()
        self.documents: List[Dict[str, Any]] = []
    
    def should_crawl(self, url: str) -> bool:
        """Check if URL should be crawled based on rules."""
        if url in self.visited:
            return False
        
        # Must be within base domain
        if not url.startswith(self.base_url):
            return False
        
        # Check exclude patterns
        for pattern in self.exclude_patterns:
            if pattern in url:
                return False
        
        # Skip media files
        if re.search(r'\.(jpg|jpeg|png|gif|pdf|zip|exe)$', url, re.I):
            return False
        
        return True
    
    def extract_content(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Extract clean content from HTML soup.
        
        Returns dict with:
        - title: Page title
        - headings: List of h1-h4 text
        - content: Clean markdown
        - links: List of internal URLs
        """
        # Remove noise elements
        for element in soup(["script", "style", "nav", "footer", "aside", "noscript", "iframe"]):
            element.decompose()
        
        # Get title
        title = soup.title.string if soup.title else url.split("/")[-1]
        title = title.strip() if title else "Untitled"
        
        # Extract headings
        headings = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3', 'h4'])]
        
        # Extract links
        links = []
        for a in soup.find_all('a', href=True):
            href = a.get('href', '')
            if not href:
                continue
            
            # Make absolute URL
            if href.startswith('/'):
                href = self.base_url + href
            elif not href.startswith('http'):
                href = urljoin(url, href)
            
            # Only include internal links
            if href.startswith(self.base_url):
                # Remove fragments and query params
                href = href.split('#')[0].split('?')[0]
                if href and href != url:
                    links.append(href)
        
        # Find main content (try different selectors for React/SPA)
        main_content = (
            soup.find('div', {'id': 'root'}) or 
            soup.find('main') or 
            soup.find('article') or 
            soup.find('div', {'class': re.compile('content|main|container', re.I)}) or
            soup.body
        )
        
        if main_content:
            content_md = md(str(main_content), heading_style="ATX")
        else:
            content_md = md(str(soup), heading_style="ATX")
        
        # Clean up markdown
        content_md = re.sub(r'You need to enable JavaScript.*?\.', '', content_md, flags=re.IGNORECASE)
        content_md = re.sub(r'\n{3,}', '\n\n', content_md)
        content_md = content_md.strip()
        
        return {
            "title": title,
            "headings": headings,
            "content": content_md,
            "links": list(set(links))
        }
    
    async def crawl_async(self, start_urls: List[str], request_delay: float = 2.0) -> List[Dict[str, Any]]:
        """
        BFS crawl with Playwright for JS rendering.
        
        Args:
            start_urls: List of seed URLs to start crawling
            request_delay: Seconds to wait between requests (politeness)
        
        Returns:
            List of document dicts with url, title, content, links, depth_level
        """
        queue = deque([(url, 0) for url in start_urls])
        
        async with async_playwright() as p:
            # Launch browser (headless mode)
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            page.set_default_timeout(30000)  # 30 seconds
            
            while queue:
                url, depth = queue.popleft()
                
                if depth > self.max_depth or not self.should_crawl(url):
                    continue
                
                try:
                    logger.info(f"🔍 [{depth}] {url}")
                    self.visited.add(url)
                    
                    # Navigate and wait for page load
                    await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                    
                    # Wait for React/SPA to render
                    try:
                        await page.wait_for_selector("body", timeout=10000)
                        await page.wait_for_timeout(3000)  # Additional wait
                        
                        # Scroll to trigger lazy loading
                        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        await page.wait_for_timeout(1000)
                    except:
                        # Fallback: just wait longer
                        await page.wait_for_timeout(5000)
                    
                    # Get rendered HTML
                    html = await page.content()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract content
                    doc_data = self.extract_content(soup, url)
                    doc_data['url'] = url
                    doc_data['depth_level'] = depth
                    
                    # Only save if content is substantial
                    if len(doc_data['content']) >= 100:
                        self.documents.append(doc_data)
                        logger.success(f"   ✅ Saved ({len(doc_data['content'])} chars, {len(doc_data['links'])} links found)")
                    else:
                        logger.warning(f"   ⚠️  Skipped (content too short: {len(doc_data['content'])} chars)")
                    
                    # Add links to queue if depth allows
                    if depth < self.max_depth:
                        links_added = 0
                        for link in doc_data['links']:
                            if link not in self.visited and link not in [item[0] for item in queue]:
                                queue.append((link, depth + 1))
                                links_added += 1
                        if links_added > 0:
                            logger.info(f"   📎 Added {links_added} new URLs to queue (depth {depth + 1})")
                    
                    # Progress update
                    logger.info(f"   📊 Progress: {len(self.documents)} docs saved, {len(self.visited)} visited, {len(queue)} in queue")
                    
                    # Polite delay
                    await asyncio.sleep(request_delay)
                    
                except Exception as e:
                    error_msg = str(e)
                    if "404" in error_msg or "net::ERR_" in error_msg:
                        logger.warning(f"   ⚠️  Page not found (404) - skipping")
                    else:
                        logger.error(f"   ❌ Error: {error_msg[:100]}")
                    continue
            
            await browser.close()
        
        return self.documents
    
    def crawl(self, start_urls: List[str], request_delay: float = 2.0) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for async crawl (for Jupyter compatibility).
        
        Args:
            start_urls: List of seed URLs
            request_delay: Seconds between requests
        
        Returns:
            List of crawled documents
        """
        return asyncio.run(self.crawl_async(start_urls, request_delay))


__all__ = ['DisputeAIWebCrawler']

