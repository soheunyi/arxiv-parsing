"""
GROBID client for the arXiv parsing system.

This module provides async HTTP client functionality for communicating with
GROBID service to extract references from PDF content with TEI XML parsing.
"""

import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import aiohttp
import aiofiles
from pydantic import BaseModel

from ..config.logging import get_logger
from ..config.settings import get_settings
from ..models.reference import Reference, ReferenceType, Venue, ParsedReference
from ..models.schemas import (
    AuthorName, DateInfo, Identifier, Location, ParserType, 
    ProcessingMetadata, ProcessingStatus
)
from .base_parser import BaseParser, ParseRequest, ParseResult, ParserCapabilities


class GrobidResponse(BaseModel):
    """Response from GROBID service."""
    
    status_code: int
    content: str
    processing_time_ms: float
    service_version: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if response indicates success."""
        return self.status_code == 200


class TEINamespaces:
    """TEI XML namespaces used by GROBID."""
    
    TEI = "http://www.tei-c.org/ns/1.0"
    XML = "http://www.w3.org/XML/1998/namespace"
    
    @classmethod
    def get_nsmap(cls) -> Dict[str, str]:
        """Get namespace map for XML parsing."""
        return {
            "tei": cls.TEI,
            "xml": cls.XML,
        }


class TEIParser:
    """Parser for TEI XML output from GROBID."""
    
    def __init__(self):
        """Initialize TEI parser."""
        self.logger = get_logger("parsers.tei_parser")
        self.namespaces = TEINamespaces.get_nsmap()
    
    def parse_references(self, tei_xml: str) -> List[Reference]:
        """
        Parse references from TEI XML.
        
        Args:
            tei_xml: TEI XML content from GROBID
            
        Returns:
            List of parsed references
        """
        try:
            root = ET.fromstring(tei_xml)
            references = []
            
            # Find bibliography section
            biblio = root.find(".//tei:listBibl", self.namespaces)
            if biblio is None:
                self.logger.warning("No bibliography section found in TEI XML")
                return references
            
            # Parse each bibliographic entry
            for i, bibl in enumerate(biblio.findall(".//tei:biblStruct", self.namespaces)):
                try:
                    reference = self._parse_bibl_struct(bibl, i)
                    if reference:
                        references.append(reference)
                except Exception as e:
                    self.logger.warning(f"Error parsing reference {i}: {e}")
                    continue
            
            self.logger.info(f"Parsed {len(references)} references from TEI XML")
            return references
            
        except ET.ParseError as e:
            self.logger.error(f"Invalid TEI XML: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error parsing TEI XML: {e}")
            return []
    
    def _parse_bibl_struct(self, bibl: ET.Element, position: int) -> Optional[Reference]:
        """
        Parse a single biblStruct element.
        
        Args:
            bibl: biblStruct XML element
            position: Position in bibliography
            
        Returns:
            Parsed reference or None if failed
        """
        reference = Reference(position_in_text=position)
        
        # Parse analytic (article) level
        analytic = bibl.find("tei:analytic", self.namespaces)
        has_analytic = analytic is not None
        if has_analytic:
            self._parse_analytic_level(analytic, reference)
        
        # Parse monogr (monograph/journal) level
        monogr = bibl.find("tei:monogr", self.namespaces)
        if monogr is not None:
            self._parse_monogr_level(monogr, reference, is_primary=not has_analytic)
        
        # Parse identifiers
        self._parse_identifiers(bibl, reference)
        
        # Parse additional elements
        self._parse_additional_elements(bibl, reference)
        
        # Determine reference type
        reference.reference_type = self._determine_reference_type(reference)
        
        return reference if reference.title or reference.authors else None
    
    def _parse_analytic_level(self, analytic: ET.Element, reference: Reference) -> None:
        """Parse analytic level (article title, authors)."""
        # Parse title
        title_elem = analytic.find("tei:title[@type='main']", self.namespaces)
        if title_elem is None:
            title_elem = analytic.find("tei:title", self.namespaces)
        
        if title_elem is not None and title_elem.text:
            reference.title = self._clean_text(title_elem.text)
        
        # Parse authors
        authors = []
        for author_elem in analytic.findall(".//tei:author", self.namespaces):
            author = self._parse_author(author_elem)
            if author:
                authors.append(author)
        
        reference.authors = authors
    
    def _parse_monogr_level(self, monogr: ET.Element, reference: Reference, is_primary: bool = False) -> None:
        """Parse monograph level (journal/book info, dates, pages).
        
        Args:
            monogr: Monograph XML element
            reference: Reference object to populate
            is_primary: If True, this is a monograph-only reference (book/report) 
                       where title and authors should be parsed as main reference info
        """
        # Parse title
        title_elem = monogr.find("tei:title[@type='main']", self.namespaces)
        if title_elem is None:
            title_elem = monogr.find("tei:title", self.namespaces)
        
        if title_elem is not None and title_elem.text:
            title_text = self._clean_text(title_elem.text)
            if is_primary:
                # For monograph-only references, this is the main title
                reference.title = title_text
                reference.venue = Venue(name="Book/Report", venue_type="book")
            else:
                # For journal articles, this is the venue name
                reference.venue = Venue(name=title_text, venue_type="journal")
        
        # Parse authors (for monograph-only references)
        if is_primary:
            authors = []
            for author_elem in monogr.findall(".//tei:author", self.namespaces):
                author = self._parse_author(author_elem)
                if author:
                    authors.append(author)
            reference.authors = authors
        
        # Parse imprint (publication details)
        imprint = monogr.find("tei:imprint", self.namespaces)
        if imprint is not None:
            self._parse_imprint(imprint, reference)
        
        # Parse publisher
        publisher_elem = monogr.find("tei:publisher", self.namespaces)
        if publisher_elem is not None and publisher_elem.text:
            reference.publisher = self._clean_text(publisher_elem.text)
    
    def _parse_imprint(self, imprint: ET.Element, reference: Reference) -> None:
        """Parse imprint element (dates, pages, volume, etc.)."""
        location = Location()
        
        # Parse date
        date_elem = imprint.find("tei:date", self.namespaces)
        if date_elem is not None:
            reference.publication_date = self._parse_date(date_elem)
        
        # Parse pages
        pages_elem = imprint.find("tei:biblScope[@unit='page']", self.namespaces)
        if pages_elem is not None:
            pages_text = pages_elem.text or ""
            location.pages = self._clean_text(pages_text)
            
            # Try to extract start and end pages
            if "--" in pages_text:
                parts = pages_text.split("--")
                location.page_start = parts[0].strip()
                location.page_end = parts[1].strip()
            elif "-" in pages_text and not pages_text.startswith("-"):
                parts = pages_text.split("-", 1)
                location.page_start = parts[0].strip()
                if len(parts) > 1:
                    location.page_end = parts[1].strip()
        
        # Parse volume
        volume_elem = imprint.find("tei:biblScope[@unit='volume']", self.namespaces)
        if volume_elem is not None and volume_elem.text:
            location.volume = self._clean_text(volume_elem.text)
        
        # Parse issue
        issue_elem = imprint.find("tei:biblScope[@unit='issue']", self.namespaces)
        if issue_elem is not None and issue_elem.text:
            location.issue = self._clean_text(issue_elem.text)
        
        reference.location = location
    
    def _parse_author(self, author_elem: ET.Element) -> Optional[AuthorName]:
        """Parse author element."""
        person = author_elem.find("tei:persName", self.namespaces)
        if person is None:
            return None
        
        # Extract name parts
        first_name = None
        middle_name = None
        last_name = None
        
        # Try structured name first
        first_elem = person.find("tei:forename[@type='first']", self.namespaces)
        if first_elem is not None and first_elem.text:
            first_name = self._clean_text(first_elem.text)
        
        middle_elem = person.find("tei:forename[@type='middle']", self.namespaces)
        if middle_elem is not None and middle_elem.text:
            middle_name = self._clean_text(middle_elem.text)
        
        surname_elem = person.find("tei:surname", self.namespaces)
        if surname_elem is not None and surname_elem.text:
            last_name = self._clean_text(surname_elem.text)
        
        # Fall back to full text if structured parsing failed
        if not any([first_name, last_name]):
            full_text = person.text or ""
            if full_text.strip():
                return AuthorName(full_name=self._clean_text(full_text))
        
        return AuthorName(
            first=first_name,
            middle=middle_name,
            last=last_name
        )
    
    def _parse_date(self, date_elem: ET.Element) -> Optional[DateInfo]:
        """Parse date element."""
        date_text = date_elem.text or ""
        when_attr = date_elem.get("when")
        
        # Prefer 'when' attribute if available
        if when_attr:
            date_text = when_attr
        
        if not date_text:
            return None
        
        date_info = DateInfo(raw_date=date_text)
        
        # Try to parse year
        year_match = re.search(r'\b(19|20)\d{2}\b', date_text)
        if year_match:
            try:
                date_info.year = int(year_match.group())
                date_info.date_confidence = 0.8
            except ValueError:
                pass
        
        return date_info
    
    def _parse_identifiers(self, bibl: ET.Element, reference: Reference) -> None:
        """Parse identifiers (DOI, arXiv, etc.)."""
        # Parse DOI
        for ptr in bibl.findall(".//tei:ptr[@type='doi']", self.namespaces):
            target = ptr.get("target")
            if target:
                # Clean DOI (remove URL prefix if present)
                doi = target.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
                reference.add_identifier("doi", doi, confidence=0.9)
        
        # Parse arXiv ID
        for ptr in bibl.findall(".//tei:ptr[@type='arxiv']", self.namespaces):
            target = ptr.get("target")
            if target:
                arxiv_id = target.replace("http://arxiv.org/abs/", "")
                reference.add_identifier("arxiv", arxiv_id, confidence=0.9)
        
        # Parse generic URLs
        for ptr in bibl.findall(".//tei:ptr", self.namespaces):
            target = ptr.get("target")
            if target and target.startswith("http") and "doi.org" not in target and "arxiv.org" not in target:
                reference.add_identifier("url", target, confidence=0.7)
    
    def _parse_additional_elements(self, bibl: ET.Element, reference: Reference) -> None:
        """Parse additional elements like notes."""
        # Parse notes
        for note in bibl.findall(".//tei:note", self.namespaces):
            if note.text:
                # Could store notes in reference metadata
                pass
    
    def _determine_reference_type(self, reference: Reference) -> ReferenceType:
        """Determine reference type based on parsed information."""
        if reference.venue and reference.venue.name:
            venue_name = reference.venue.name.lower()
            
            # Check for conference indicators
            conference_indicators = [
                "proceedings", "conference", "symposium", "workshop",
                "meeting", "congress", "summit", "forum"
            ]
            if any(indicator in venue_name for indicator in conference_indicators):
                return ReferenceType.CONFERENCE_PAPER
            
            # Check for journal indicators
            journal_indicators = [
                "journal", "review", "letters", "communications",
                "transactions", "annals", "bulletin"
            ]
            if any(indicator in venue_name for indicator in journal_indicators):
                return ReferenceType.JOURNAL_ARTICLE
        
        # Check for book indicators
        if reference.publisher:
            return ReferenceType.BOOK
        
        # Check for preprint indicators
        if reference.arxiv_id:
            return ReferenceType.PREPRINT
        
        # Default to journal article
        return ReferenceType.JOURNAL_ARTICLE
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common artifacts
        text = text.replace('\n', ' ').replace('\t', ' ')
        
        return text


class GrobidClient(BaseParser):
    """
    GROBID client for reference extraction from PDF files.
    
    This parser uses the GROBID service to extract bibliographic references
    from PDF documents and parse the resulting TEI XML.
    """
    
    def __init__(self, **config: Any):
        """Initialize GROBID client."""
        super().__init__(**config)
        self.settings = get_settings()
        self.tei_parser = TEIParser()
        
        # HTTP session will be created when needed
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Service health
        self._service_available = None
        self._last_health_check = None
    
    @property
    def parser_type(self) -> ParserType:
        """Return parser type."""
        return ParserType.GROBID
    
    @property
    def capabilities(self) -> ParserCapabilities:
        """Return parser capabilities."""
        return ParserCapabilities(
            parser_type=ParserType.GROBID,
            supported_content_types=["pdf"],
            supports_async=True,
            supports_batch=False,
            confidence_scoring=True,
            field_level_extraction=True,
            requires_external_service=True,
            service_dependencies=["grobid"],
            max_file_size_mb=50,
            max_processing_time_seconds=300,
            quality_features={
                "author_extraction": True,
                "title_extraction": True,
                "venue_extraction": True,
                "date_extraction": True,
                "doi_extraction": True,
                "citation_context": False,
            }
        )
    
    async def initialize(self) -> None:
        """Initialize GROBID client and check service availability."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.settings.grobid_timeout)
            connector = aiohttp.TCPConnector(
                limit=self.settings.grobid_max_connections,
                limit_per_host=self.settings.grobid_max_connections
            )
            
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
        
        # Check service health
        await self._check_service_health()
        
        self.logger.info("GROBID client initialized")
    
    async def cleanup(self) -> None:
        """Clean up HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
        
        self.logger.info("GROBID client cleaned up")
    
    async def _check_service_health(self) -> bool:
        """Check if GROBID service is available."""
        try:
            url = f"{self.settings.grobid_url}/api/isalive"
            async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                self._service_available = response.status == 200
                self._last_health_check = asyncio.get_event_loop().time()
                
                if self._service_available:
                    self.logger.info("GROBID service is available")
                else:
                    self.logger.warning(f"GROBID service returned status {response.status}")
                
                return self._service_available
                
        except Exception as e:
            self._service_available = False
            self._last_health_check = asyncio.get_event_loop().time()
            self.logger.error(f"GROBID service health check failed: {e}")
            return False
    
    async def _parse_implementation(self, request: ParseRequest) -> ParseResult:
        """Core GROBID parsing implementation."""
        # Ensure service is available
        if not self._service_available:
            await self._check_service_health()
        
        if not self._service_available:
            raise Exception("GROBID service is not available")
        
        # Handle file path vs content
        if isinstance(request.content, Path):
            pdf_path = request.content
        else:
            # Save content to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(request.content.encode() if isinstance(request.content, str) else request.content)
                pdf_path = Path(tmp.name)
        
        try:
            # Call GROBID service
            grobid_response = await self._call_grobid_service(pdf_path)
            
            if not grobid_response.success:
                raise Exception(f"GROBID service error: HTTP {grobid_response.status_code}")
            
            # Parse TEI XML response
            references = self.tei_parser.parse_references(grobid_response.content)
            
            # Add parsed versions to references
            for ref in references:
                parsed_ref = ParsedReference(
                    parser_type=ParserType.GROBID,
                    raw_text=grobid_response.content,
                    parsed_fields={
                        "title": ref.title,
                        "authors": [str(author) for author in ref.authors],
                        "venue": ref.venue_name,
                        "year": ref.publication_year,
                        "doi": ref.doi,
                    },
                    processing_metadata=ProcessingMetadata(
                        parser_type=ParserType.GROBID,
                        processing_time_ms=grobid_response.processing_time_ms,
                        confidence_score=self._calculate_confidence(ref)
                    ),
                    confidence_scores=self._calculate_field_confidences(ref)
                )
                ref.add_parsed_version(parsed_ref)
            
            # Create result
            metadata = ProcessingMetadata(
                parser_type=ParserType.GROBID,
                processing_time_ms=grobid_response.processing_time_ms,
                confidence_score=self._calculate_overall_confidence(references)
            )
            
            return ParseResult(
                references=references,
                processing_metadata=metadata,
                raw_output=grobid_response.content
            )
            
        finally:
            # Clean up temporary file if created
            if isinstance(request.content, str) and pdf_path.exists():
                pdf_path.unlink()
    
    async def _call_grobid_service(self, pdf_path: Path) -> GrobidResponse:
        """Call GROBID service to extract references."""
        url = f"{self.settings.grobid_url}/api/processReferences"
        
        # Prepare multipart form data
        data = aiohttp.FormData()
        
        async with aiofiles.open(pdf_path, 'rb') as f:
            pdf_content = await f.read()
        
        data.add_field(
            'input',
            pdf_content,
            filename=pdf_path.name,
            content_type='application/pdf'
        )
        
        # Additional GROBID parameters
        data.add_field('consolidateHeader', '1')
        data.add_field('consolidateCitations', '1')
        data.add_field('includeRawCitations', '1')
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with self._session.post(url, data=data) as response:
                processing_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                content = await response.text()
                
                return GrobidResponse(
                    status_code=response.status,
                    content=content,
                    processing_time_ms=processing_time_ms,
                    service_version=response.headers.get('X-GROBID-Version')
                )
                
        except Exception as e:
            processing_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            raise Exception(f"GROBID service call failed: {e}")
    
    def _calculate_confidence(self, reference: Reference) -> float:
        """Calculate confidence score for a reference."""
        score = 0.0
        
        # Title: 30%
        if reference.title:
            score += 0.3
        
        # Authors: 25%
        if reference.authors:
            score += 0.25
        
        # Venue: 20%
        if reference.venue_name:
            score += 0.2
        
        # Date: 15%
        if reference.publication_year:
            score += 0.15
        
        # Identifiers: 10%
        if reference.doi or reference.arxiv_id:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_field_confidences(self, reference: Reference) -> Dict[str, float]:
        """Calculate confidence scores for individual fields."""
        confidences = {}
        
        if reference.title:
            # Higher confidence for longer, well-formed titles
            title_score = min(len(reference.title) / 50.0, 1.0) * 0.8 + 0.2
            confidences["title"] = title_score
        
        if reference.authors:
            # Higher confidence for more complete author information
            author_score = sum(
                1.0 if author.last and author.first else 0.5
                for author in reference.authors
            ) / len(reference.authors)
            confidences["authors"] = author_score
        
        if reference.venue_name:
            confidences["venue"] = 0.8  # GROBID is generally good at venues
        
        if reference.publication_year:
            confidences["year"] = 0.9  # Years are usually accurate
        
        if reference.doi:
            confidences["doi"] = 0.95  # DOIs are very reliable
        
        return confidences
    
    def _calculate_overall_confidence(self, references: List[Reference]) -> float:
        """Calculate overall confidence for the parsing result."""
        if not references:
            return 0.0
        
        individual_scores = [
            self._calculate_confidence(ref) for ref in references
        ]
        
        return sum(individual_scores) / len(individual_scores)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on GROBID service."""
        base_health = await super().health_check()
        
        # Add GROBID-specific health information
        service_healthy = await self._check_service_health()
        
        base_health.update({
            "grobid_service_url": self.settings.grobid_url,
            "grobid_service_available": service_healthy,
            "last_health_check": self._last_health_check,
        })
        
        if not service_healthy:
            base_health["status"] = "unhealthy"
            base_health["error"] = "GROBID service is not available"
        
        return base_health