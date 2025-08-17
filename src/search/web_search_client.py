#!/usr/bin/env python3
"""
Free Web Search Client for arXiv paper discovery.

This module provides unlimited search capabilities without API limitations by:
1. Direct web scraping of search engines
2. Multiple search engine support (Google, Bing, DuckDuckGo)
3. arXiv-specific search patterns
4. Rate limiting and respectful scraping

No API keys required - completely free and unlimited.
"""

import asyncio
import re
import urllib.parse
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup
from pydantic import BaseModel

from src.config.settings import get_settings
from src.models.reference import Reference
from src.utils.logging import get_logger


class SearchResult(BaseModel):
    """Search result from web scraping."""
    title: str
    url: str
    snippet: str
    source_engine: str  # google, bing, duckduckgo
    is_arxiv: bool = False
    arxiv_id: Optional[str] = None
    confidence_score: float = 0.0  # How likely this is the correct paper


class SearchEngine:
    """Base class for search engine scrapers."""
    
    def __init__(self, name: str, base_url: str):
        self.name = name
        self.base_url = base_url
        self.logger = get_logger(f"search.{name}")
    
    def construct_search_url(self, query: str) -> str:
        """Construct search URL for the engine."""
        raise NotImplementedError
    
    async def parse_results(self, html: str) -> List[SearchResult]:
        """Parse search results from HTML."""
        raise NotImplementedError


class GoogleSearchEngine(SearchEngine):
    """Google search engine scraper."""
    
    def __init__(self):
        super().__init__("google", "https://www.google.com")
    
    def construct_search_url(self, query: str, num_results: int = 20) -> str:
        """Construct Google search URL."""
        encoded_query = urllib.parse.quote_plus(query)
        return f"{self.base_url}/search?q={encoded_query}&num={num_results}&hl=en"
    
    async def parse_results(self, html: str) -> List[SearchResult]:
        """Parse Google search results."""
        results = []
        soup = BeautifulSoup(html, 'html.parser')
        
        # Google search result selectors (may need updates as Google changes)
        result_containers = soup.find_all('div', class_='g')
        
        for container in result_containers:
            try:
                # Extract title and URL
                title_elem = container.find('h3')
                if not title_elem:
                    continue
                
                title = title_elem.get_text(strip=True)
                
                # Find the link
                link_elem = container.find('a', href=True)
                if not link_elem:
                    continue
                
                url = link_elem['href']
                if url.startswith('/url?q='):
                    # Extract actual URL from Google redirect
                    url = urllib.parse.parse_qs(url[7:]).get('q', [''])[0]
                
                # Extract snippet
                snippet_elem = container.find('span', class_=['VwiC3b', 'aCOpRe'])
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                
                # Check if this is an arXiv paper
                is_arxiv, arxiv_id = self._detect_arxiv_paper(url, title, snippet)
                
                result = SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    source_engine=self.name,
                    is_arxiv=is_arxiv,
                    arxiv_id=arxiv_id,
                    confidence_score=self._calculate_confidence(title, snippet, is_arxiv)
                )
                results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Error parsing Google result: {e}")
                continue
        
        return results
    
    def _detect_arxiv_paper(self, url: str, title: str, snippet: str) -> Tuple[bool, Optional[str]]:
        """Detect if a search result is an arXiv paper."""
        # Check URL patterns
        arxiv_url_pattern = re.compile(r'arxiv\.org/abs/(\d{4}\.\d{4,5})')
        match = arxiv_url_pattern.search(url)
        if match:
            return True, match.group(1)
        
        # Check title/snippet for arXiv ID
        arxiv_id_pattern = re.compile(r'arXiv:(\d{4}\.\d{4,5})')
        for text in [title, snippet]:
            match = arxiv_id_pattern.search(text)
            if match:
                return True, match.group(1)
        
        # Check if URL contains arxiv domain
        if 'arxiv.org' in url.lower():
            return True, None
        
        return False, None
    
    def _calculate_confidence(self, title: str, snippet: str, is_arxiv: bool) -> float:
        """Calculate confidence score for the result."""
        score = 0.5  # Base score
        
        if is_arxiv:
            score += 0.3  # Boost for arXiv papers
        
        # Length and quality indicators
        if len(title) > 20:
            score += 0.1
        
        if len(snippet) > 50:
            score += 0.1
        
        # Academic indicators in snippet
        academic_terms = ['paper', 'research', 'study', 'analysis', 'method', 'algorithm']
        snippet_lower = snippet.lower()
        academic_score = sum(0.02 for term in academic_terms if term in snippet_lower)
        score += min(academic_score, 0.1)
        
        return min(score, 1.0)


class BingSearchEngine(SearchEngine):
    """Bing search engine scraper."""
    
    def __init__(self):
        super().__init__("bing", "https://www.bing.com")
    
    def construct_search_url(self, query: str, num_results: int = 20) -> str:
        """Construct Bing search URL."""
        encoded_query = urllib.parse.quote_plus(query)
        return f"{self.base_url}/search?q={encoded_query}&count={num_results}"
    
    async def parse_results(self, html: str) -> List[SearchResult]:
        """Parse Bing search results."""
        results = []
        soup = BeautifulSoup(html, 'html.parser')
        
        # Bing search result selectors
        result_containers = soup.find_all('li', class_='b_algo')
        
        for container in result_containers:
            try:
                # Extract title and URL
                title_elem = container.find('h2')
                if not title_elem:
                    continue
                
                link_elem = title_elem.find('a', href=True)
                if not link_elem:
                    continue
                
                title = link_elem.get_text(strip=True)
                url = link_elem['href']
                
                # Extract snippet
                snippet_elem = container.find('p')
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                
                # Check if this is an arXiv paper
                is_arxiv, arxiv_id = self._detect_arxiv_paper(url, title, snippet)
                
                result = SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    source_engine=self.name,
                    is_arxiv=is_arxiv,
                    arxiv_id=arxiv_id,
                    confidence_score=self._calculate_confidence(title, snippet, is_arxiv)
                )
                results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Error parsing Bing result: {e}")
                continue
        
        return results
    
    def _detect_arxiv_paper(self, url: str, title: str, snippet: str) -> Tuple[bool, Optional[str]]:
        """Detect arXiv papers (same logic as Google)."""
        arxiv_url_pattern = re.compile(r'arxiv\.org/abs/(\d{4}\.\d{4,5})')
        match = arxiv_url_pattern.search(url)
        if match:
            return True, match.group(1)
        
        arxiv_id_pattern = re.compile(r'arXiv:(\d{4}\.\d{4,5})')
        for text in [title, snippet]:
            match = arxiv_id_pattern.search(text)
            if match:
                return True, match.group(1)
        
        if 'arxiv.org' in url.lower():
            return True, None
        
        return False, None
    
    def _calculate_confidence(self, title: str, snippet: str, is_arxiv: bool) -> float:
        """Calculate confidence score."""
        score = 0.5
        if is_arxiv:
            score += 0.3
        if len(title) > 20:
            score += 0.1
        if len(snippet) > 50:
            score += 0.1
        return min(score, 1.0)


class DuckDuckGoSearchEngine(SearchEngine):
    """DuckDuckGo search engine scraper."""
    
    def __init__(self):
        super().__init__("duckduckgo", "https://duckduckgo.com")
    
    def construct_search_url(self, query: str, num_results: int = 20) -> str:
        """Construct DuckDuckGo search URL."""
        encoded_query = urllib.parse.quote_plus(query)
        return f"{self.base_url}/?q={encoded_query}"
    
    async def parse_results(self, html: str) -> List[SearchResult]:
        """Parse DuckDuckGo search results."""
        results = []
        soup = BeautifulSoup(html, 'html.parser')
        
        # DuckDuckGo search result selectors
        result_containers = soup.find_all('div', class_='result')
        
        for container in result_containers:
            try:
                # Extract title and URL
                title_elem = container.find('h2', class_='result__title')
                if not title_elem:
                    continue
                
                link_elem = title_elem.find('a', href=True)
                if not link_elem:
                    continue
                
                title = link_elem.get_text(strip=True)
                url = link_elem['href']
                
                # Extract snippet
                snippet_elem = container.find('a', class_='result__snippet')
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                
                # Check if this is an arXiv paper
                is_arxiv, arxiv_id = self._detect_arxiv_paper(url, title, snippet)
                
                result = SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    source_engine=self.name,
                    is_arxiv=is_arxiv,
                    arxiv_id=arxiv_id,
                    confidence_score=self._calculate_confidence(title, snippet, is_arxiv)
                )
                results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Error parsing DuckDuckGo result: {e}")
                continue
        
        return results
    
    def _detect_arxiv_paper(self, url: str, title: str, snippet: str) -> Tuple[bool, Optional[str]]:
        """Detect arXiv papers."""
        arxiv_url_pattern = re.compile(r'arxiv\.org/abs/(\d{4}\.\d{4,5})')
        match = arxiv_url_pattern.search(url)
        if match:
            return True, match.group(1)
        
        arxiv_id_pattern = re.compile(r'arXiv:(\d{4}\.\d{4,5})')
        for text in [title, snippet]:
            match = arxiv_id_pattern.search(text)
            if match:
                return True, match.group(1)
        
        if 'arxiv.org' in url.lower():
            return True, None
        
        return False, None
    
    def _calculate_confidence(self, title: str, snippet: str, is_arxiv: bool) -> float:
        """Calculate confidence score."""
        score = 0.5
        if is_arxiv:
            score += 0.3
        if len(title) > 20:
            score += 0.1
        if len(snippet) > 50:
            score += 0.1
        return min(score, 1.0)


class WebSearchClient:
    """
    Free web search client with unlimited searches.
    
    Features:
    - No API keys required
    - Multiple search engines (Google, Bing, DuckDuckGo)
    - Automatic arXiv paper detection
    - Rate limiting for respectful scraping
    - Result deduplication and ranking
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger("search.web_client")
        
        # Initialize search engines
        self.engines = [
            GoogleSearchEngine(),
            BingSearchEngine(),
            DuckDuckGoSearchEngine()
        ]
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting (respectful scraping)
        self.request_delay = 1.0  # 1 second between requests
        self.last_request_time = {}
        
        # User agent rotation for better success rate
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0"
        ]
        self.current_user_agent_index = 0
    
    async def initialize(self):
        """Initialize the web search client."""
        if not self.session or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(total=30)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                }
            )
        
        self.logger.info("Web search client initialized")
    
    async def cleanup(self):
        """Clean up resources."""
        if self.session and not self.session.closed:
            await self.session.close()
        self.logger.info("Web search client cleaned up")
    
    async def search_for_reference(self, reference: Reference, max_results: int = 10) -> List[SearchResult]:
        """
        Search for a specific reference across multiple search engines.
        
        Args:
            reference: Parsed reference from GROBID
            max_results: Maximum number of results to return
            
        Returns:
            List of deduplicated and ranked search results
        """
        # Construct search query
        query = self._construct_search_query(reference)
        if not query:
            self.logger.warning("Could not construct search query for reference")
            return []
        
        self.logger.info(f"Searching for: {query}")
        
        # Search across multiple engines
        all_results = []
        for engine in self.engines:
            try:
                engine_results = await self._search_engine(engine, query, max_results)
                all_results.extend(engine_results)
                
                # Respectful delay between engines
                await asyncio.sleep(self.request_delay)
                
            except Exception as e:
                self.logger.warning(f"Search failed on {engine.name}: {e}")
                continue
        
        # Deduplicate and rank results
        deduplicated_results = self._deduplicate_results(all_results)
        ranked_results = self._rank_results(deduplicated_results, reference)
        
        return ranked_results[:max_results]
    
    async def _search_engine(self, engine: SearchEngine, query: str, max_results: int) -> List[SearchResult]:
        """Search a specific engine."""
        await self._respect_rate_limit(engine.name)
        
        search_url = engine.construct_search_url(query, max_results)
        
        # Rotate user agent
        headers = {
            'User-Agent': self.user_agents[self.current_user_agent_index]
        }
        self.current_user_agent_index = (self.current_user_agent_index + 1) % len(self.user_agents)
        
        try:
            async with self.session.get(search_url, headers=headers) as response:
                if response.status != 200:
                    self.logger.warning(f"Search request failed with status {response.status}")
                    return []
                
                html = await response.text()
                results = await engine.parse_results(html)
                
                self.logger.info(f"Found {len(results)} results from {engine.name}")
                return results
                
        except Exception as e:
            self.logger.error(f"Error searching {engine.name}: {e}")
            return []
    
    async def _respect_rate_limit(self, engine_name: str):
        """Implement rate limiting for respectful scraping."""
        now = datetime.now().timestamp()
        last_request = self.last_request_time.get(engine_name, 0)
        
        time_since_last = now - last_request
        if time_since_last < self.request_delay:
            sleep_time = self.request_delay - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time[engine_name] = datetime.now().timestamp()
    
    def _construct_search_query(self, reference: Reference) -> str:
        """Construct search query from reference data."""
        query_parts = []
        
        # Title is most important - use exact match
        if reference.title:
            # Clean title and add quotes for exact matching
            clean_title = re.sub(r'[^\w\s\-]', ' ', reference.title)
            clean_title = ' '.join(clean_title.split())  # Normalize whitespace
            query_parts.append(f'"{clean_title}"')
        
        # Add first author for better precision
        if reference.authors and len(reference.authors) > 0:
            first_author = reference.authors[0]
            if first_author.last:
                query_parts.append(f'"{first_author.last}"')
        
        # Add year if available
        if reference.publication_year:
            query_parts.append(str(reference.publication_year))
        
        # Add arXiv-specific search terms
        query_parts.append("arxiv OR \"arXiv:\" OR site:arxiv.org")
        
        return " ".join(query_parts)
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on URL and title similarity."""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            # Normalize URL for comparison
            normalized_url = self._normalize_url(result.url)
            
            if normalized_url not in seen_urls:
                seen_urls.add(normalized_url)
                unique_results.append(result)
        
        return unique_results
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication."""
        # Remove common variations
        url = url.lower()
        url = re.sub(r'[?&]utm_[^&]*', '', url)  # Remove UTM parameters
        url = re.sub(r'[?&]ref=[^&]*', '', url)   # Remove ref parameters
        url = re.sub(r'/$', '', url)              # Remove trailing slash
        
        return url
    
    def _rank_results(self, results: List[SearchResult], reference: Reference) -> List[SearchResult]:
        """Rank results by relevance to the original reference."""
        for result in results:
            # Start with base confidence score
            score = result.confidence_score
            
            # Boost arXiv papers significantly
            if result.is_arxiv:
                score += 0.4
            
            # Title similarity boost
            if reference.title and result.title:
                title_similarity = self._calculate_title_similarity(reference.title, result.title)
                score += title_similarity * 0.3
            
            # Author matching boost
            if reference.authors and result.snippet:
                author_match_score = self._calculate_author_match(reference.authors, result.snippet)
                score += author_match_score * 0.2
            
            # Year matching boost
            if reference.publication_year and str(reference.publication_year) in result.snippet:
                score += 0.1
            
            result.confidence_score = min(score, 1.0)
        
        # Sort by confidence score (descending)
        return sorted(results, key=lambda r: r.confidence_score, reverse=True)
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles."""
        if not title1 or not title2:
            return 0.0
        
        # Normalize titles
        def normalize_title(title):
            # Remove punctuation, convert to lowercase, split into words
            clean = re.sub(r'[^\w\s]', ' ', title.lower())
            return set(clean.split())
        
        words1 = normalize_title(title1)
        words2 = normalize_title(title2)
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_author_match(self, authors: List, snippet: str) -> float:
        """Calculate author matching score."""
        if not authors or not snippet:
            return 0.0
        
        snippet_lower = snippet.lower()
        matches = 0
        
        for author in authors[:3]:  # Check first 3 authors
            if hasattr(author, 'last') and author.last:
                if author.last.lower() in snippet_lower:
                    matches += 1
        
        return min(matches / len(authors[:3]), 1.0) if authors else 0.0


# Example usage and testing
async def test_web_search():
    """Test the web search functionality."""
    client = WebSearchClient()
    await client.initialize()
    
    try:
        # Create a test reference
        from src.models.reference import Reference, AuthorName
        
        test_ref = Reference(
            title="Attention Is All You Need",
            authors=[AuthorName(first="Ashish", last="Vaswani")],
            publication_year=2017
        )
        
        # Search for the reference
        results = await client.search_for_reference(test_ref, max_results=10)
        
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.title}")
            print(f"   URL: {result.url}")
            print(f"   Engine: {result.source_engine}")
            print(f"   arXiv: {result.is_arxiv} (ID: {result.arxiv_id})")
            print(f"   Confidence: {result.confidence_score:.3f}")
            print()
    
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(test_web_search())