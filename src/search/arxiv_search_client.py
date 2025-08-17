#!/usr/bin/env python3
"""
arXiv Search Client for direct paper discovery.

This module provides direct search in arXiv database for:
1. Finding papers by title, authors, or keywords
2. Validating if extracted references exist in arXiv
3. Unlimited searches using arXiv API
4. Advanced search filters (category, date range, etc.)

Uses arXiv API - completely free, no rate limits for reasonable usage.
"""

import asyncio
import re
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import aiohttp
from pydantic import BaseModel

from src.config.settings import get_settings
from src.models.reference import Reference, AuthorName
from src.config.logging import get_logger


class ArxivPaper(BaseModel):
    """arXiv paper model."""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published_date: datetime
    updated_date: datetime
    pdf_url: str
    abs_url: str
    doi: Optional[str] = None
    journal_ref: Optional[str] = None
    confidence_score: float = 0.0  # How well this matches the search query


class ArxivSearchResult(BaseModel):
    """arXiv search result container."""
    query: str
    total_results: int
    papers: List[ArxivPaper]
    search_time_ms: float
    found_matching_paper: bool = False


class ArxivSearchClient:
    """
    Direct arXiv search client.
    
    Features:
    - Free unlimited searches via arXiv API
    - Advanced search by title, author, category, date
    - Paper existence validation
    - Fuzzy matching for reference validation
    - Batch search capabilities
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger("search.arxiv")
        
        # arXiv API configuration
        self.api_base_url = "http://export.arxiv.org/api/query"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting (arXiv requests 3 seconds between calls)
        self.request_delay = 3.1  # Slightly more than 3 seconds to be safe
        self.last_request_time = 0
        
        # Search configuration
        self.max_results_per_query = 100  # arXiv API limit
        self.default_results = 20
    
    async def initialize(self):
        """Initialize the arXiv search client."""
        if not self.session or self.session.closed:
            connector = aiohttp.TCPConnector(limit=10)
            timeout = aiohttp.ClientTimeout(total=60)  # arXiv can be slow
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'User-Agent': 'arXiv-Parser/1.0 (Research Tool; https://github.com/example/arxiv-parser)'
                }
            )
        
        self.logger.info("arXiv search client initialized")
    
    async def cleanup(self):
        """Clean up resources."""
        if self.session and not self.session.closed:
            await self.session.close()
        self.logger.info("arXiv search client cleaned up")
    
    async def search_for_reference(self, reference: Reference, max_results: int = 10) -> ArxivSearchResult:
        """
        Search arXiv for a specific reference.
        
        Args:
            reference: Parsed reference from GROBID
            max_results: Maximum number of results to return
            
        Returns:
            ArxivSearchResult with matching papers
        """
        start_time = datetime.now()
        
        # Try different search strategies
        search_strategies = [
            self._search_by_title_and_author,
            self._search_by_title_only,
            self._search_by_author_and_keywords,
        ]
        
        all_papers = []
        best_match = None
        
        for strategy in search_strategies:
            try:
                query, papers = await strategy(reference, max_results)
                if papers:
                    all_papers.extend(papers)
                    
                    # Check if we found a high-confidence match
                    high_conf_papers = [p for p in papers if p.confidence_score > 0.8]
                    if high_conf_papers and not best_match:
                        best_match = high_conf_papers[0]
                        break  # Stop if we found a very good match
                        
            except Exception as e:
                self.logger.warning(f"Search strategy failed: {e}")
                continue
        
        # Deduplicate and rank results
        unique_papers = self._deduplicate_papers(all_papers)
        ranked_papers = self._rank_papers(unique_papers, reference)
        
        search_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        return ArxivSearchResult(
            query=self._construct_basic_query(reference),
            total_results=len(ranked_papers),
            papers=ranked_papers[:max_results],
            search_time_ms=search_time_ms,
            found_matching_paper=len(ranked_papers) > 0 and ranked_papers[0].confidence_score > 0.6
        )
    
    async def _search_by_title_and_author(self, reference: Reference, max_results: int) -> Tuple[str, List[ArxivPaper]]:
        """Search by combining title and first author."""
        if not reference.title or not reference.authors:
            return "", []
        
        # Construct query with title and first author
        title_keywords = self._extract_title_keywords(reference.title)
        first_author = reference.authors[0]
        
        query_parts = []
        
        # Add title keywords
        if title_keywords:
            title_query = " AND ".join(f'ti:"{keyword}"' for keyword in title_keywords[:3])  # Top 3 keywords
            query_parts.append(f"({title_query})")
        
        # Add multiple authors (up to 3 for better matching)
        author_queries = []
        for author in reference.authors[:3]:  # Use up to 3 authors
            if author.last:
                author_queries.append(f'au:"{author.last}"')
        
        if author_queries:
            # Use OR for multiple authors (any author match is good)
            if len(author_queries) == 1:
                query_parts.append(author_queries[0])
            else:
                author_query = "(" + " OR ".join(author_queries) + ")"
                query_parts.append(author_query)
        
        query = " AND ".join(query_parts)
        
        papers = await self._execute_search(query, max_results)
        return query, papers
    
    async def _search_by_title_only(self, reference: Reference, max_results: int) -> Tuple[str, List[ArxivPaper]]:
        """Search by title keywords only."""
        if not reference.title:
            return "", []
        
        title_keywords = self._extract_title_keywords(reference.title)
        if not title_keywords:
            return "", []
        
        # Try exact title first
        exact_title_query = f'ti:"{reference.title}"'
        papers = await self._execute_search(exact_title_query, min(max_results, 5))
        
        # If no exact matches, try keyword search
        if not papers:
            keyword_query = " AND ".join(f'ti:"{keyword}"' for keyword in title_keywords[:4])
            papers = await self._execute_search(keyword_query, max_results)
        
        query = exact_title_query if papers else keyword_query
        return query, papers
    
    async def _search_by_author_and_keywords(self, reference: Reference, max_results: int) -> Tuple[str, List[ArxivPaper]]:
        """Search by author and extracted keywords."""
        if not reference.authors:
            return "", []
        
        first_author = reference.authors[0]
        if not first_author.last:
            return "", []
        
        query_parts = []
        
        # Add multiple authors (up to 4 for comprehensive matching)
        author_queries = []
        for author in reference.authors[:4]:  # Use up to 4 authors
            if author.last:
                author_queries.append(f'au:"{author.last}"')
        
        if author_queries:
            # Use OR for multiple authors (any author match is good)
            if len(author_queries) == 1:
                query_parts.append(author_queries[0])
            else:
                author_query = "(" + " OR ".join(author_queries) + ")"
                query_parts.append(author_query)
        
        # Add title keywords if available
        if reference.title:
            title_keywords = self._extract_title_keywords(reference.title)
            if title_keywords:
                # Use broader search in title or abstract
                keyword_query = " OR ".join(f'ti:"{keyword}" OR abs:"{keyword}"' for keyword in title_keywords[:2])
                query_parts.append(f"({keyword_query})")
        
        # Add year filter if available
        if reference.publication_year:
            year = reference.publication_year
            # Search within ±1 year range
            start_date = f"{year-1}0101"
            end_date = f"{year+1}1231"
            query_parts.append(f"submittedDate:[{start_date} TO {end_date}]")
        
        query = " AND ".join(query_parts)
        papers = await self._execute_search(query, max_results)
        
        return query, papers
    
    async def _execute_search(self, query: str, max_results: int) -> List[ArxivPaper]:
        """Execute arXiv API search."""
        await self._respect_rate_limit()
        
        # Construct API URL
        params = {
            'search_query': query,
            'start': 0,
            'max_results': min(max_results, self.max_results_per_query),
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        url = f"{self.api_base_url}?" + "&".join(f"{k}={quote(str(v))}" for k, v in params.items())
        
        self.logger.debug(f"arXiv API query: {query}")
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    self.logger.warning(f"arXiv API request failed with status {response.status}")
                    return []
                
                xml_content = await response.text()
                papers = self._parse_arxiv_response(xml_content)
                
                self.logger.info(f"Found {len(papers)} papers for query: {query[:100]}...")
                return papers
                
        except Exception as e:
            self.logger.error(f"Error executing arXiv search: {e}")
            return []
    
    async def _respect_rate_limit(self):
        """Implement rate limiting for arXiv API."""
        now = datetime.now().timestamp()
        time_since_last = now - self.last_request_time
        
        if time_since_last < self.request_delay:
            sleep_time = self.request_delay - time_since_last
            self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.1f} seconds")
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = datetime.now().timestamp()
    
    def _parse_arxiv_response(self, xml_content: str) -> List[ArxivPaper]:
        """Parse arXiv API XML response."""
        papers = []
        
        try:
            # Parse XML
            root = ET.fromstring(xml_content)
            
            # Define namespaces
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            # Find all entry elements
            entries = root.findall('atom:entry', namespaces)
            
            for entry in entries:
                try:
                    paper = self._parse_entry(entry, namespaces)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    self.logger.warning(f"Error parsing arXiv entry: {e}")
                    continue
                    
        except ET.ParseError as e:
            self.logger.error(f"Error parsing arXiv XML response: {e}")
        
        return papers
    
    def _parse_entry(self, entry: ET.Element, namespaces: Dict[str, str]) -> Optional[ArxivPaper]:
        """Parse a single arXiv entry."""
        try:
            # Extract arXiv ID
            id_elem = entry.find('atom:id', namespaces)
            if id_elem is None:
                return None
            
            full_id = id_elem.text
            arxiv_id = full_id.split('/')[-1]  # Extract ID from URL
            
            # Extract title
            title_elem = entry.find('atom:title', namespaces)
            if title_elem is not None and title_elem.text:
                title = title_elem.text.strip()
                # Clean up any newlines or extra whitespace
                title = ' '.join(title.split())
            else:
                title = ""
            
            # Extract authors
            authors = []
            author_elems = entry.findall('atom:author', namespaces)
            for author_elem in author_elems:
                name_elem = author_elem.find('atom:name', namespaces)
                if name_elem is not None:
                    authors.append(name_elem.text.strip())
            
            # Extract abstract
            summary_elem = entry.find('atom:summary', namespaces)
            abstract = summary_elem.text.strip() if summary_elem is not None else ""
            
            # Extract categories
            categories = []
            category_elems = entry.findall('atom:category', namespaces)
            for cat_elem in category_elems:
                term = cat_elem.get('term')
                if term:
                    categories.append(term)
            
            # Extract dates
            published_elem = entry.find('atom:published', namespaces)
            if published_elem is not None:
                published_date = datetime.fromisoformat(published_elem.text.replace('Z', '+00:00')).replace(tzinfo=None)
            else:
                published_date = datetime.now()
            
            updated_elem = entry.find('atom:updated', namespaces)
            if updated_elem is not None:
                updated_date = datetime.fromisoformat(updated_elem.text.replace('Z', '+00:00')).replace(tzinfo=None)
            else:
                updated_date = published_date
            
            # Extract links
            pdf_url = ""
            abs_url = ""
            link_elems = entry.findall('atom:link', namespaces)
            for link_elem in link_elems:
                href = link_elem.get('href', '')
                link_title = link_elem.get('title', '')  # Changed variable name
                if 'pdf' in link_title.lower():
                    pdf_url = href
                elif href and 'abs' in href:
                    abs_url = href
            
            # Extract DOI and journal reference
            doi = None
            journal_ref = None
            
            # Look for DOI in arXiv namespace
            doi_elem = entry.find('arxiv:doi', namespaces)
            if doi_elem is not None:
                doi = doi_elem.text.strip()
            
            # Look for journal reference
            journal_elem = entry.find('arxiv:journal_ref', namespaces)
            if journal_elem is not None:
                journal_ref = journal_elem.text.strip()
            
            return ArxivPaper(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                categories=categories,
                published_date=published_date,
                updated_date=updated_date,
                pdf_url=pdf_url,
                abs_url=abs_url,
                doi=doi,
                journal_ref=journal_ref
            )
            
        except Exception as e:
            self.logger.warning(f"Error parsing arXiv entry: {e}")
            return None
    
    def _extract_title_keywords(self, title: str) -> List[str]:
        """Extract meaningful keywords from title."""
        if not title:
            return []
        
        # Remove common stop words and short words
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'via', 'using', 'through', 'towards'
        }
        
        # Clean title and extract words
        clean_title = re.sub(r'[^\w\s-]', ' ', title.lower())
        words = clean_title.split()
        
        # Filter meaningful keywords
        keywords = []
        for word in words:
            if (len(word) >= 3 and 
                word not in stop_words and 
                not word.isdigit()):
                keywords.append(word)
        
        return keywords[:6]  # Return top 6 keywords
    
    def _construct_basic_query(self, reference: Reference) -> str:
        """Construct a basic search query for logging."""
        parts = []
        if reference.title:
            parts.append(f"title:{reference.title[:30]}...")
        if reference.authors:
            parts.append(f"author:{reference.authors[0].last}")
        if reference.publication_year:
            parts.append(f"year:{reference.publication_year}")
        return " ".join(parts)
    
    def _deduplicate_papers(self, papers: List[ArxivPaper]) -> List[ArxivPaper]:
        """Remove duplicate papers by arXiv ID."""
        seen_ids = set()
        unique_papers = []
        
        for paper in papers:
            if paper.arxiv_id not in seen_ids:
                seen_ids.add(paper.arxiv_id)
                unique_papers.append(paper)
        
        return unique_papers
    
    def _rank_papers(self, papers: List[ArxivPaper], reference: Reference) -> List[ArxivPaper]:
        """Rank papers by relevance to the reference."""
        for paper in papers:
            score = 0.0
            
            # Title similarity (most important)
            if reference.title and paper.title:
                title_sim = self._calculate_title_similarity(reference.title, paper.title)
                score += title_sim * 0.5
            
            # Author matching
            if reference.authors:
                author_score = self._calculate_author_match(reference.authors, paper.authors)
                score += author_score * 0.3
            
            # Year proximity
            if reference.publication_year:
                year_score = self._calculate_year_proximity(reference.publication_year, paper.published_date.year)
                score += year_score * 0.1
            
            # Exact title match gets significant boost
            if (reference.title and paper.title and 
                self._normalize_text(reference.title) == self._normalize_text(paper.title)):
                score += 0.3
            
            # Recent papers get slight boost
            days_old = (datetime.now() - paper.published_date).days
            if days_old < 30:
                score += 0.05
            
            paper.confidence_score = min(score, 1.0)
        
        return sorted(papers, key=lambda p: p.confidence_score, reverse=True)
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate title similarity using word overlap."""
        if not title1 or not title2:
            return 0.0
        
        words1 = set(self._normalize_text(title1).split())
        words2 = set(self._normalize_text(title2).split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _calculate_author_match(self, ref_authors: List[AuthorName], paper_authors: List[str]) -> float:
        """Calculate author matching score."""
        if not ref_authors or not paper_authors:
            return 0.0
        
        matches = 0
        paper_authors_lower = [author.lower() for author in paper_authors]
        
        for ref_author in ref_authors[:3]:  # Check first 3 authors
            if ref_author.last:
                last_name = ref_author.last.lower()
                # Check if last name appears in any paper author
                if any(last_name in paper_author for paper_author in paper_authors_lower):
                    matches += 1
        
        return matches / min(len(ref_authors), 3)
    
    def _calculate_year_proximity(self, ref_year: int, paper_year: int) -> float:
        """Calculate year proximity score."""
        year_diff = abs(ref_year - paper_year)
        if year_diff == 0:
            return 1.0
        elif year_diff == 1:
            return 0.8
        elif year_diff <= 2:
            return 0.5
        else:
            return 0.0
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Remove punctuation, convert to lowercase, normalize whitespace
        normalized = re.sub(r'[^\w\s]', ' ', text.lower())
        return ' '.join(normalized.split())
    
    async def search_by_exact_title(self, title: str, max_results: int = 5) -> List[ArxivPaper]:
        """Search for papers with exact title match."""
        query = f'ti:"{title}"'
        papers = await self._execute_search(query, max_results)
        return papers
    
    async def search_by_author(self, author_name: str, max_results: int = 20) -> List[ArxivPaper]:
        """Search for papers by author name."""
        query = f'au:"{author_name}"'
        papers = await self._execute_search(query, max_results)
        return papers
    
    async def search_by_arxiv_id(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """Search for a specific paper by arXiv ID."""
        query = f'id:{arxiv_id}'
        papers = await self._execute_search(query, 1)
        return papers[0] if papers else None
    
    async def search_by_category(self, category: str, max_results: int = 50) -> List[ArxivPaper]:
        """Search for papers in a specific category."""
        query = f'cat:{category}'
        papers = await self._execute_search(query, max_results)
        return papers


# Example usage and testing functions
async def test_arxiv_search():
    """Test the arXiv search functionality with various examples."""
    client = ArxivSearchClient()
    await client.initialize()
    
    try:
        print("🔍 Testing arXiv Search Client")
        print("=" * 50)
        
        # Test 1: Search by exact title
        print("\n1. Testing exact title search:")
        papers = await client.search_by_exact_title("Attention Is All You Need")
        print(f"Found {len(papers)} papers with exact title match")
        if papers:
            paper = papers[0]
            print(f"   - {paper.title}")
            print(f"   - arXiv ID: {paper.arxiv_id}")
            print(f"   - Authors: {', '.join(paper.authors[:3])}")
        
        # Test 2: Search by author
        print("\n2. Testing author search:")
        papers = await client.search_by_author("Yoshua Bengio", max_results=5)
        print(f"Found {len(papers)} papers by Yoshua Bengio")
        for i, paper in enumerate(papers[:3], 1):
            print(f"   {i}. {paper.title[:60]}... ({paper.arxiv_id})")
        
        # Test 3: Search for reference
        print("\n3. Testing reference search:")
        from src.models.reference import Reference, AuthorName
        
        test_ref = Reference(
            title="Deep Residual Learning for Image Recognition",
            authors=[AuthorName(first="Kaiming", last="He")],
            publication_year=2015
        )
        
        result = await client.search_for_reference(test_ref, max_results=5)
        print(f"Reference search results:")
        print(f"   - Query: {result.query}")
        print(f"   - Found matching paper: {result.found_matching_paper}")
        print(f"   - Total results: {result.total_results}")
        print(f"   - Search time: {result.search_time_ms:.0f}ms")
        
        if result.papers:
            best_match = result.papers[0]
            print(f"   Best match:")
            print(f"     - Title: {best_match.title}")
            print(f"     - arXiv ID: {best_match.arxiv_id}")
            print(f"     - Confidence: {best_match.confidence_score:.3f}")
            print(f"     - Authors: {', '.join(best_match.authors[:2])}")
        
        # Test 4: Search by category
        print("\n4. Testing category search:")
        papers = await client.search_by_category("cs.LG", max_results=3)
        print(f"Found {len(papers)} papers in cs.LG category:")
        for i, paper in enumerate(papers, 1):
            print(f"   {i}. {paper.title[:50]}... ({paper.arxiv_id})")
        
        # Test 5: Search by arXiv ID
        print("\n5. Testing arXiv ID search:")
        paper = await client.search_by_arxiv_id("1706.03762")
        if paper:
            print(f"Found paper by ID:")
            print(f"   - Title: {paper.title}")
            print(f"   - Authors: {', '.join(paper.authors[:3])}")
            print(f"   - Categories: {', '.join(paper.categories)}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
    
    finally:
        await client.cleanup()


async def search_for_specific_paper(title: str, author_last_name: str = None, year: int = None):
    """
    Example function to search for a specific paper.
    
    Args:
        title: Paper title
        author_last_name: Last name of first author
        year: Publication year
    """
    client = ArxivSearchClient()
    await client.initialize()
    
    try:
        # Create a reference object
        from src.models.reference import Reference, AuthorName
        
        authors = [AuthorName(last=author_last_name)] if author_last_name else []
        
        reference = Reference(
            title=title,
            authors=authors,
            publication_year=year
        )
        
        # Search for the reference
        result = await client.search_for_reference(reference, max_results=10)
        
        print(f"Search Results for: {title}")
        print("=" * 60)
        print(f"Query executed: {result.query}")
        print(f"Search time: {result.search_time_ms:.0f}ms")
        print(f"Found matching paper: {'✅' if result.found_matching_paper else '❌'}")
        print(f"Total results: {result.total_results}")
        print()
        
        if result.papers:
            print("Top results:")
            for i, paper in enumerate(result.papers[:5], 1):
                print(f"{i}. {paper.title}")
                print(f"   arXiv ID: {paper.arxiv_id}")
                print(f"   Authors: {', '.join(paper.authors[:3])}")
                print(f"   Confidence: {paper.confidence_score:.3f}")
                print(f"   Published: {paper.published_date.strftime('%Y-%m-%d')}")
                print()
        else:
            print("No papers found matching the criteria.")
        
        return result
        
    finally:
        await client.cleanup()


if __name__ == "__main__":
    # Run basic tests
    print("Running arXiv search tests...")
    asyncio.run(test_arxiv_search())
    
    print("\n" + "="*60)
    print("Example: Search for specific paper")
    
    # Example: Search for a specific paper
    asyncio.run(search_for_specific_paper(
        title="Attention Is All You Need",
        author_last_name="Vaswani",
        year=2017
    ))