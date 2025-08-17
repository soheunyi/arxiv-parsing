"""
ArXiv content fetcher for the arXiv parsing system.

This module provides async HTTP client functionality for fetching arXiv papers
in various formats (HTML, PDF, metadata) with rate limiting and error handling.
"""

import asyncio
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import aiohttp
import aiofiles
from pydantic import BaseModel

from ..config.logging import get_logger
from ..config.settings import get_settings
from ..models.paper import ArxivCategory, PaperMetadata, PaperContent
from ..models.schemas import ContentType, DateInfo, AuthorName


class ArxivFetchError(Exception):
    """Custom exception for arXiv fetching errors."""
    pass


class RateLimiter:
    """Rate limiter for arXiv API requests."""
    
    def __init__(self, delay_seconds: float = 3.1):
        """
        Initialize rate limiter.
        
        Args:
            delay_seconds: Minimum delay between requests
        """
        self.delay_seconds = delay_seconds
        self.last_request_time: Optional[datetime] = None
        self._lock = asyncio.Lock()
    
    async def wait(self) -> None:
        """Wait for rate limit if necessary."""
        async with self._lock:
            if self.last_request_time is not None:
                elapsed = (datetime.utcnow() - self.last_request_time).total_seconds()
                if elapsed < self.delay_seconds:
                    wait_time = self.delay_seconds - elapsed
                    await asyncio.sleep(wait_time)
            
            self.last_request_time = datetime.utcnow()


class ArxivMetadata(BaseModel):
    """Raw arXiv metadata from API response."""
    
    id: str
    title: str
    authors: List[str]
    summary: str
    categories: List[str]
    primary_category: str
    published: str
    updated: str
    doi: Optional[str] = None
    journal_ref: Optional[str] = None
    comments: Optional[str] = None
    
    @classmethod
    def from_atom_entry(cls, entry_dict: Dict) -> "ArxivMetadata":
        """Create from Atom feed entry."""
        # Extract arXiv ID from URL
        arxiv_url = entry_dict.get("id", "")
        arxiv_id = arxiv_url.split("/")[-1] if arxiv_url else ""
        
        # Extract authors
        authors = []
        for author in entry_dict.get("authors", []):
            if isinstance(author, dict):
                authors.append(author.get("name", ""))
            else:
                authors.append(str(author))
        
        # Extract categories
        categories = []
        for cat in entry_dict.get("categories", []):
            if isinstance(cat, dict):
                categories.append(cat.get("term", ""))
            else:
                categories.append(str(cat))
        
        return cls(
            id=arxiv_id,
            title=entry_dict.get("title", "").strip(),
            authors=authors,
            summary=entry_dict.get("summary", "").strip(),
            categories=categories,
            primary_category=entry_dict.get("primary_category", {}).get("term", ""),
            published=entry_dict.get("published", ""),
            updated=entry_dict.get("updated", ""),
            doi=entry_dict.get("doi"),
            journal_ref=entry_dict.get("journal_ref"),
            comments=entry_dict.get("comment"),
        )


class ArxivFetcher:
    """
    Async HTTP client for fetching arXiv content.
    
    Provides methods to fetch papers in various formats with proper rate limiting,
    caching, and error handling.
    """
    
    def __init__(self):
        """Initialize the arXiv fetcher."""
        self.settings = get_settings()
        self.logger = get_logger("ingestion.arxiv_fetcher")
        
        # Rate limiting
        self.rate_limiter = RateLimiter(self.settings.arxiv_rate_limit_delay)
        
        # Cache directory
        self.cache_dir = self.settings.cache_dir / "arxiv"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Session will be created when needed
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Statistics
        self._stats = {
            "requests_made": 0,
            "cache_hits": 0,
            "errors": 0,
            "rate_limited": 0,
        }
    
    async def __aenter__(self) -> "ArxivFetcher":
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_session(self) -> None:
        """Ensure HTTP session is created."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.settings.arxiv_timeout)
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            
            headers = {
                "User-Agent": "arXiv-Parser/1.0 (Research Tool)"
            }
            
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers=headers
            )
    
    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _normalize_arxiv_id(self, arxiv_id: str) -> str:
        """
        Normalize arXiv ID format.
        
        Args:
            arxiv_id: Raw arXiv ID
            
        Returns:
            Normalized arXiv ID
        """
        # Remove common prefixes
        arxiv_id = arxiv_id.strip()
        if arxiv_id.lower().startswith("arxiv:"):
            arxiv_id = arxiv_id[6:]
        if arxiv_id.startswith("http"):
            # Extract ID from URL
            arxiv_id = arxiv_id.split("/")[-1]
        
        # Remove version if present for caching purposes
        if "v" in arxiv_id and arxiv_id[-2:].isdigit():
            base_id = arxiv_id.rsplit("v", 1)[0]
            version = int(arxiv_id.rsplit("v", 1)[1])
        else:
            base_id = arxiv_id
            version = 1
        
        return base_id
    
    def _get_cache_path(self, arxiv_id: str, content_type: str) -> Path:
        """Get cache file path for content."""
        safe_id = re.sub(r'[^\w.-]', '_', arxiv_id)
        extension = {
            "html": ".html",
            "pdf": ".pdf",
            "metadata": ".json",
            "source": ".tar.gz"
        }.get(content_type, ".dat")
        
        return self.cache_dir / f"{safe_id}_{content_type}{extension}"
    
    async def _cache_content(self, arxiv_id: str, content_type: str, content: bytes) -> None:
        """Cache content to disk."""
        try:
            cache_path = self._get_cache_path(arxiv_id, content_type)
            async with aiofiles.open(cache_path, "wb") as f:
                await f.write(content)
            
            self.logger.debug(f"Cached {content_type} for {arxiv_id}")
        except Exception as e:
            self.logger.warning(f"Failed to cache content: {e}")
    
    async def _load_from_cache(self, arxiv_id: str, content_type: str) -> Optional[bytes]:
        """Load content from cache if available and fresh."""
        try:
            cache_path = self._get_cache_path(arxiv_id, content_type)
            
            if not cache_path.exists():
                return None
            
            # Check if cache is fresh (24 hours)
            cache_age = datetime.utcnow().timestamp() - cache_path.stat().st_mtime
            if cache_age > 86400:  # 24 hours
                self.logger.debug(f"Cache expired for {arxiv_id} {content_type}")
                return None
            
            async with aiofiles.open(cache_path, "rb") as f:
                content = await f.read()
            
            self._stats["cache_hits"] += 1
            self.logger.debug(f"Cache hit for {arxiv_id} {content_type}")
            return content
            
        except Exception as e:
            self.logger.warning(f"Failed to load from cache: {e}")
            return None
    
    async def fetch_metadata(self, arxiv_id: str) -> PaperMetadata:
        """
        Fetch paper metadata from arXiv API.
        
        Args:
            arxiv_id: arXiv paper ID
            
        Returns:
            Paper metadata
            
        Raises:
            ArxivFetchError: If fetching fails
        """
        arxiv_id = self._normalize_arxiv_id(arxiv_id)
        
        # Check cache first
        cached_content = await self._load_from_cache(arxiv_id, "metadata")
        if cached_content:
            # Parse cached JSON metadata
            import orjson
            try:
                metadata_dict = orjson.loads(cached_content)
                return PaperMetadata(**metadata_dict)
            except Exception as e:
                self.logger.warning(f"Failed to parse cached metadata: {e}")
        
        await self._ensure_session()
        await self.rate_limiter.wait()
        
        # Fetch from arXiv API
        api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        
        try:
            self._stats["requests_made"] += 1
            
            async with self._session.get(api_url) as response:
                if response.status != 200:
                    raise ArxivFetchError(f"HTTP {response.status}: {await response.text()}")
                
                content = await response.text()
                
                # Parse Atom XML (simplified)
                metadata = await self._parse_atom_response(content, arxiv_id)
                
                # Cache the metadata
                import orjson
                await self._cache_content(
                    arxiv_id, 
                    "metadata", 
                    orjson.dumps(metadata.dict())
                )
                
                return metadata
                
        except aiohttp.ClientError as e:
            self._stats["errors"] += 1
            raise ArxivFetchError(f"Network error fetching metadata: {e}")
        except Exception as e:
            self._stats["errors"] += 1
            raise ArxivFetchError(f"Error fetching metadata: {e}")
    
    async def _parse_atom_response(self, xml_content: str, arxiv_id: str) -> PaperMetadata:
        """Parse Atom XML response from arXiv API (simplified version)."""
        # This is a simplified parser - in a real implementation,
        # you'd use a proper XML parser like lxml
        
        # For now, create a basic metadata object
        # In practice, you'd parse the XML properly
        
        lines = xml_content.split('\n')
        title = ""
        authors = []
        summary = ""
        categories = []
        published = ""
        updated = ""
        
        # Very basic parsing - would need proper XML parsing in production
        for line in lines:
            line = line.strip()
            if "<title>" in line and "</title>" in line:
                title = line.split("<title>")[1].split("</title>")[0].strip()
            elif "<name>" in line and "</name>" in line:
                author_name = line.split("<name>")[1].split("</name>")[0].strip()
                if author_name:
                    authors.append(AuthorName(full_name=author_name))
            elif "<summary>" in line and "</summary>" in line:
                summary = line.split("<summary>")[1].split("</summary>")[0].strip()
            elif "<published>" in line and "</published>" in line:
                published = line.split("<published>")[1].split("</published>")[0].strip()
            elif "<updated>" in line and "</updated>" in line:
                updated = line.split("<updated>")[1].split("</updated>")[0].strip()
        
        # Parse dates
        submission_date = None
        if published:
            try:
                pub_dt = datetime.fromisoformat(published.replace('Z', '+00:00'))
                submission_date = DateInfo(
                    raw_date=published,
                    parsed_date=pub_dt,
                    year=pub_dt.year,
                    month=pub_dt.month,
                    day=pub_dt.day,
                    date_confidence=0.9
                )
            except:
                pass
        
        last_updated = None
        if updated:
            try:
                upd_dt = datetime.fromisoformat(updated.replace('Z', '+00:00'))
                last_updated = DateInfo(
                    raw_date=updated,
                    parsed_date=upd_dt,
                    year=upd_dt.year,
                    month=upd_dt.month,
                    day=upd_dt.day,
                    date_confidence=0.9
                )
            except:
                pass
        
        # Create categories
        categories_obj = None
        if categories:
            categories_obj = ArxivCategory(
                primary=categories[0] if categories else "unknown",
                secondary=categories[1:] if len(categories) > 1 else []
            )
        
        return PaperMetadata(
            arxiv_id=arxiv_id,
            title=title or f"Paper {arxiv_id}",
            authors=authors,
            submission_date=submission_date,
            last_updated=last_updated,
            categories=categories_obj,
        )
    
    async def fetch_html(self, arxiv_id: str) -> Optional[str]:
        """
        Fetch HTML version of paper.
        
        Args:
            arxiv_id: arXiv paper ID
            
        Returns:
            HTML content or None if not available
        """
        arxiv_id = self._normalize_arxiv_id(arxiv_id)
        
        # Check cache
        cached_content = await self._load_from_cache(arxiv_id, "html")
        if cached_content:
            return cached_content.decode('utf-8')
        
        await self._ensure_session()
        await self.rate_limiter.wait()
        
        # Try HTML version (not all papers have this)
        html_url = f"{self.settings.arxiv_base_url}/html/{arxiv_id}"
        
        try:
            self._stats["requests_made"] += 1
            
            async with self._session.get(html_url) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # Cache the content
                    await self._cache_content(arxiv_id, "html", content.encode('utf-8'))
                    
                    self.logger.info(f"Successfully fetched HTML for {arxiv_id}")
                    return content
                elif response.status == 404:
                    self.logger.info(f"HTML not available for {arxiv_id}")
                    return None
                else:
                    self.logger.warning(f"HTTP {response.status} fetching HTML for {arxiv_id}")
                    return None
                    
        except Exception as e:
            self.logger.warning(f"Error fetching HTML for {arxiv_id}: {e}")
            return None
    
    async def fetch_pdf(self, arxiv_id: str, save_path: Optional[Path] = None) -> Optional[Path]:
        """
        Fetch PDF version of paper.
        
        Args:
            arxiv_id: arXiv paper ID
            save_path: Optional path to save PDF
            
        Returns:
            Path to saved PDF file or None if failed
        """
        arxiv_id = self._normalize_arxiv_id(arxiv_id)
        
        if save_path is None:
            save_path = self._get_cache_path(arxiv_id, "pdf")
        
        # Check if already cached
        if save_path.exists():
            cache_age = datetime.utcnow().timestamp() - save_path.stat().st_mtime
            if cache_age < 86400:  # 24 hours
                self._stats["cache_hits"] += 1
                return save_path
        
        await self._ensure_session()
        await self.rate_limiter.wait()
        
        pdf_url = f"{self.settings.arxiv_base_url}/pdf/{arxiv_id}.pdf"
        
        try:
            self._stats["requests_made"] += 1
            
            async with self._session.get(pdf_url) as response:
                if response.status == 200:
                    # Ensure directory exists
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Save PDF content
                    async with aiofiles.open(save_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                    
                    self.logger.info(f"Successfully fetched PDF for {arxiv_id}")
                    return save_path
                else:
                    self.logger.error(f"HTTP {response.status} fetching PDF for {arxiv_id}")
                    return None
                    
        except Exception as e:
            self._stats["errors"] += 1
            self.logger.error(f"Error fetching PDF for {arxiv_id}: {e}")
            return None
    
    async def fetch_paper_content(self, arxiv_id: str) -> Tuple[PaperMetadata, Optional[PaperContent]]:
        """
        Fetch complete paper content (metadata + HTML/PDF).
        
        Args:
            arxiv_id: arXiv paper ID
            
        Returns:
            Tuple of (metadata, content)
        """
        arxiv_id = self._normalize_arxiv_id(arxiv_id)
        
        # Fetch metadata first
        metadata = await self.fetch_metadata(arxiv_id)
        
        # Try to fetch HTML first
        html_content = await self.fetch_html(arxiv_id)
        
        content = None
        if html_content:
            content = PaperContent(
                title=metadata.title,
                html_content=html_content,
                content_type=ContentType.HTML,
                language="en"
            )
            
            # Extract abstract from HTML if possible
            # This would require proper HTML parsing
            content.abstract = "Abstract extraction from HTML would be implemented here"
        
        # If no HTML, try PDF as fallback
        elif self.settings.enable_pdf_fallback:
            pdf_path = await self.fetch_pdf(arxiv_id)
            if pdf_path:
                content = PaperContent(
                    title=metadata.title,
                    pdf_path=pdf_path,
                    content_type=ContentType.PDF,
                    language="en"
                )
        
        return metadata, content
    
    def get_statistics(self) -> Dict[str, int]:
        """Get fetcher statistics."""
        return self._stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset statistics."""
        self._stats = {
            "requests_made": 0,
            "cache_hits": 0,
            "errors": 0,
            "rate_limited": 0,
        }