"""
arXiv HTML parser for the arXiv parsing system.

This module provides parsing functionality for arXiv HTML documents,
extracting bibliographic references from the structured HTML format.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel

from ..config.logging import get_logger
from ..models.reference import Reference, ReferenceType, Venue, ParsedReference
from ..models.schemas import (
    AuthorName, DateInfo, Identifier, Location, ParserType, 
    ProcessingMetadata, ProcessingStatus
)
from .base_parser import BaseParser, ParseRequest, ParseResult, ParserCapabilities


class ArxivHtmlParser(BaseParser):
    """
    arXiv HTML parser for reference extraction from arXiv HTML documents.
    
    This parser extracts bibliographic references from arXiv HTML papers
    using the structured HTML format with ltx_bibitem classes.
    """
    
    def __init__(self, **config: Any):
        """Initialize arXiv HTML parser."""
        super().__init__(**config)
        
    @property
    def parser_type(self) -> ParserType:
        """Return parser type."""
        return ParserType.HTML
    
    @property
    def capabilities(self) -> ParserCapabilities:
        """Return parser capabilities."""
        return ParserCapabilities(
            parser_type=ParserType.HTML,
            supported_content_types=["html"],
            supports_async=True,
            supports_batch=False,
            confidence_scoring=True,
            field_level_extraction=True,
            requires_external_service=False,
            service_dependencies=[],
            max_file_size_mb=10,
            max_processing_time_seconds=60,
            quality_features={
                "author_extraction": True,
                "title_extraction": True,
                "venue_extraction": True,
                "date_extraction": True,
                "doi_extraction": False,  # arXiv HTML doesn't typically include DOIs
                "citation_context": False,
            }
        )
    
    async def initialize(self) -> None:
        """Initialize arXiv HTML parser."""
        self.logger.info("arXiv HTML parser initialized")
    
    async def cleanup(self) -> None:
        """Clean up parser resources."""
        self.logger.info("arXiv HTML parser cleaned up")
    
    async def _parse_implementation(self, request: ParseRequest) -> ParseResult:
        """Core arXiv HTML parsing implementation."""
        # Load HTML content
        if isinstance(request.content, Path):
            with open(request.content, 'r', encoding='utf-8') as f:
                html_content = f.read()
        else:
            html_content = request.content
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract references
        references = self._extract_references(soup)
        
        # Add parsed versions to references
        for ref in references:
            parsed_ref = ParsedReference(
                parser_type=ParserType.HTML,
                raw_text=html_content,
                parsed_fields={
                    "title": ref.title,
                    "authors": [str(author) for author in ref.authors],
                    "venue": ref.venue_name,
                    "year": ref.publication_year,
                    "doi": ref.doi,
                },
                processing_metadata=ProcessingMetadata(
                    parser_type=ParserType.HTML,
                    confidence_score=self._calculate_confidence(ref)
                ),
                confidence_scores=self._calculate_field_confidences(ref)
            )
            ref.add_parsed_version(parsed_ref)
        
        # Create result
        metadata = ProcessingMetadata(
            parser_type=ParserType.HTML,
            confidence_score=self._calculate_overall_confidence(references)
        )
        
        return ParseResult(
            references=references,
            processing_metadata=metadata,
            raw_output=html_content
        )
    
    def _extract_references(self, soup: BeautifulSoup) -> List[Reference]:
        """Extract references from arXiv HTML."""
        references = []
        
        # Find bibliography section
        bib_section = soup.find('section', class_='ltx_bibliography')
        if not bib_section:
            self.logger.warning("No bibliography section found in HTML")
            return references
        
        # Find bibliography list
        bib_list = bib_section.find('ul', class_='ltx_biblist')
        if not bib_list:
            self.logger.warning("No bibliography list found in HTML")
            return references
        
        # Extract bibliography items
        bib_items = bib_list.find_all('li', class_='ltx_bibitem')
        self.logger.info(f"Found {len(bib_items)} bibliography items")
        
        for i, item in enumerate(bib_items):
            try:
                reference = self._parse_bibliography_item(item, i)
                if reference:
                    references.append(reference)
            except Exception as e:
                self.logger.warning(f"Error parsing reference {i}: {e}")
                continue
        
        self.logger.info(f"Successfully parsed {len(references)} references")
        return references
    
    def _parse_bibliography_item(self, item: Tag, position: int) -> Optional[Reference]:
        """Parse a single bibliography item."""
        reference = Reference(position_in_text=position)
        
        # Get item ID
        item_id = item.get('id', '')
        if item_id:
            reference.add_identifier('arxiv_bib_id', item_id, confidence=1.0)
        
        # Get reference number
        tag_elem = item.find('span', class_='ltx_tag')
        if tag_elem:
            tag_text = tag_elem.get_text().strip()
            # Extract number from [1], [2], etc.
            number_match = re.search(r'\[(\d+)\]', tag_text)
            if number_match:
                reference.position_in_text = int(number_match.group(1)) - 1
        
        # Parse bibliography blocks (can be multiple)
        bibblocks = item.find_all('span', class_='ltx_bibblock')
        if not bibblocks:
            return None
        
        # Try structured parsing first (with formatting classes)
        self._parse_structured_reference(bibblocks, reference)
        
        # If structured parsing didn't work, try unstructured parsing
        if not reference.title and not reference.authors:
            self._parse_unstructured_reference(bibblocks, reference)
        
        # Determine reference type
        reference.reference_type = self._determine_reference_type(reference)
        
        return reference if reference.title or reference.authors else None
    
    def _parse_structured_reference(self, bibblocks: List[Tag], reference: Reference) -> None:
        """Parse reference using structured formatting (smallcaps, italic)."""
        # Combine all bibblocks for structured parsing
        combined_block = BeautifulSoup('', 'html.parser')
        combined_span = combined_block.new_tag('span')
        
        for block in bibblocks:
            # Copy contents to combined block
            for child in block.children:
                if hasattr(child, 'name'):
                    combined_span.append(child.extract())
                else:
                    combined_span.append(str(child))
        
        # Parse using existing methods
        self._parse_authors(combined_span, reference)
        self._parse_title(combined_span, reference)
        self._parse_venue_and_details(combined_span, reference)
    
    def _parse_unstructured_reference(self, bibblocks: List[Tag], reference: Reference) -> None:
        """Parse reference from unstructured text (no formatting classes)."""
        # Extract text from all blocks
        text_parts = []
        for block in bibblocks:
            text = block.get_text().strip()
            if text:
                text_parts.append(text)
        
        if not text_parts:
            return
        
        # Common patterns for unstructured references:
        # 1. "Authors. Title. Venue, Year."
        # 2. "Authors (Year). Title. Venue."
        # 3. "Authors Title Venue Year"
        
        # Try to identify parts based on position and patterns
        if len(text_parts) >= 2:
            # First block is likely authors
            authors_text = text_parts[0].rstrip('.')
            reference.authors = self._parse_author_names(authors_text)
            
            # Second block is likely title  
            title_text = text_parts[1].rstrip('.')
            if title_text:
                reference.title = title_text
            
            # Remaining blocks are venue/details
            if len(text_parts) > 2:
                venue_text = ' '.join(text_parts[2:])
                self._parse_publication_details(venue_text, reference)
        elif len(text_parts) == 1:
            # Single block - try to parse as full citation
            full_text = text_parts[0]
            self._parse_full_citation_text(full_text, reference)
    
    def _parse_full_citation_text(self, citation_text: str, reference: Reference) -> None:
        """Parse a full citation from a single text block."""
        # Look for common patterns in the full text
        
        # Try to extract year first (helps identify structure)
        year_match = re.search(r'\b(19|20)\d{2}\b', citation_text)
        if year_match:
            year = int(year_match.group())
            reference.publication_date = DateInfo(
                year=year,
                raw_date=year_match.group(),
                date_confidence=0.8
            )
        
        # Pattern 1: "Authors. Title. Venue, Year."
        period_split = citation_text.split('.')
        if len(period_split) >= 3:
            # First part likely authors
            authors_text = period_split[0].strip()
            if authors_text and not authors_text.lower().startswith(('the', 'a', 'an')):
                reference.authors = self._parse_author_names(authors_text)
            
            # Second part likely title
            title_text = period_split[1].strip()
            if title_text:
                reference.title = title_text
            
            # Rest is venue/details
            venue_text = '.'.join(period_split[2:]).strip()
            if venue_text:
                self._parse_publication_details(venue_text, reference)
        else:
            # Try comma-based splitting as fallback
            comma_split = citation_text.split(',')
            if len(comma_split) >= 2:
                # Look for patterns like "Author1, Author2 and Author3"
                potential_authors = comma_split[0].strip()
                rest = ','.join(comma_split[1:]).strip()
                
                # Simple heuristic: if first part contains "and" or common name patterns
                if ' and ' in potential_authors or re.search(r'\b[A-Z]\.\s*[A-Z]', potential_authors):
                    reference.authors = self._parse_author_names(potential_authors)
                    
                    # Try to extract title from the rest
                    # Look for sentence-case text that could be a title
                    title_match = re.search(r'([A-Z][^.]*[a-z][^.]*)', rest)
                    if title_match:
                        reference.title = title_match.group(1).strip()
                    
                    # Parse remaining as venue details
                    self._parse_publication_details(rest, reference)
    
    def _parse_authors(self, bibblock: Tag, reference: Reference) -> None:
        """Parse authors from bibliography block."""
        # Authors are typically in spans with ltx_font_smallcaps
        author_spans = bibblock.find_all('span', class_='ltx_font_smallcaps')
        
        if author_spans:
            # Take the first span as authors (there might be multiple for different purposes)
            author_text = author_spans[0].get_text().strip()
            
            # Parse author names
            authors = self._parse_author_names(author_text)
            reference.authors = authors
    
    def _parse_author_names(self, author_text: str) -> List[AuthorName]:
        """Parse individual author names from author text."""
        authors = []
        
        # Split by 'and' and commas, handling various formats
        # Common patterns: "A. Smith and B. Jones", "Smith, A. and Jones, B."
        
        # First, split by 'and'
        parts = re.split(r'\s+and\s+', author_text, flags=re.IGNORECASE)
        
        for part in parts:
            part = part.strip().rstrip(',')
            if not part:
                continue
            
            # Try to parse name structure
            author = self._parse_single_author(part)
            if author:
                authors.append(author)
        
        return authors
    
    def _parse_single_author(self, name_text: str) -> Optional[AuthorName]:
        """Parse a single author name."""
        name_text = name_text.strip()
        if not name_text:
            return None
        
        # Remove common suffixes/prefixes
        name_text = re.sub(r'\s+(Jr\.?|Sr\.?|Ph\.?D\.?|M\.?D\.?)$', '', name_text, flags=re.IGNORECASE)
        
        # Check for "Last, First Middle" format
        if ',' in name_text:
            parts = name_text.split(',', 1)
            last_name = parts[0].strip()
            rest = parts[1].strip()
            
            # Split rest into first and middle names
            name_parts = rest.split()
            first_name = name_parts[0] if name_parts else None
            middle_name = ' '.join(name_parts[1:]) if len(name_parts) > 1 else None
            
            return AuthorName(
                first=first_name,
                middle=middle_name or None,
                last=last_name
            )
        else:
            # "First Middle Last" format
            name_parts = name_text.split()
            if len(name_parts) == 1:
                # Only one name, assume it's the last name
                return AuthorName(last=name_parts[0])
            elif len(name_parts) == 2:
                # First Last
                return AuthorName(first=name_parts[0], last=name_parts[1])
            else:
                # First Middle... Last
                first_name = name_parts[0]
                last_name = name_parts[-1]
                middle_name = ' '.join(name_parts[1:-1]) if len(name_parts) > 2 else None
                
                return AuthorName(
                    first=first_name,
                    middle=middle_name,
                    last=last_name
                )
    
    def _parse_title(self, bibblock: Tag, reference: Reference) -> None:
        """Parse title from bibliography block."""
        # Title is typically in spans with ltx_font_italic
        title_spans = bibblock.find_all('span', class_='ltx_font_italic')
        
        if title_spans:
            # Take the first italic span as title
            title_text = title_spans[0].get_text().strip()
            
            # Clean title (remove trailing commas, normalize whitespace)
            title_text = re.sub(r'\s+', ' ', title_text.strip().rstrip(','))
            
            if title_text:
                reference.title = title_text
    
    def _parse_venue_and_details(self, bibblock: Tag, reference: Reference) -> None:
        """Parse venue, year, pages, and other details."""
        # Get full text and remove structured parts (authors, title)
        full_text = bibblock.get_text()
        
        # Remove author text
        author_spans = bibblock.find_all('span', class_='ltx_font_smallcaps')
        for span in author_spans:
            full_text = full_text.replace(span.get_text(), '', 1)
        
        # Remove title text
        title_spans = bibblock.find_all('span', class_='ltx_font_italic')
        for span in title_spans:
            full_text = full_text.replace(span.get_text(), '', 1)
        
        # Clean remaining text
        remaining_text = re.sub(r'\s+', ' ', full_text.strip()).strip(',').strip()
        
        if remaining_text:
            self._parse_publication_details(remaining_text, reference)
    
    def _parse_publication_details(self, details_text: str, reference: Reference) -> None:
        """Parse publication details from remaining text."""
        # Look for year pattern
        year_match = re.search(r'\b(19|20)\d{2}\b', details_text)
        if year_match:
            try:
                year = int(year_match.group())
                reference.publication_date = DateInfo(
                    year=year,
                    raw_date=year_match.group(),
                    date_confidence=0.9
                )
            except ValueError:
                pass
        
        # Look for page patterns
        page_patterns = [
            r'pp?\.\s*(\d+(?:[-–]\d+)?)',  # pp. 123-456
            r'pages?\s*(\d+(?:[-–]\d+)?)',  # pages 123-456
            r'\b(\d+(?:[-–]\d+))\s*$',  # 123-456 at end
        ]
        
        location = Location()
        for pattern in page_patterns:
            page_match = re.search(pattern, details_text, re.IGNORECASE)
            if page_match:
                pages = page_match.group(1)
                location.pages = pages
                
                # Try to extract start and end pages
                if '–' in pages or '-' in pages:
                    page_parts = re.split(r'[-–]', pages)
                    if len(page_parts) == 2:
                        location.page_start = page_parts[0].strip()
                        location.page_end = page_parts[1].strip()
                break
        
        # Look for volume pattern
        volume_match = re.search(r'\bvol(?:ume)?\s*(\d+)', details_text, re.IGNORECASE)
        if volume_match:
            location.volume = volume_match.group(1)
        
        # Extract venue (everything before year, after removing pages)
        venue_text = details_text
        if year_match:
            venue_text = details_text[:year_match.start()].strip()
        
        # Remove page information from venue
        for pattern in page_patterns:
            venue_text = re.sub(pattern, '', venue_text, flags=re.IGNORECASE)
        
        venue_text = venue_text.strip().rstrip(',').strip()
        
        # Clean up venue text
        venue_text = re.sub(r'\s+', ' ', venue_text)
        venue_text = venue_text.strip('.,')
        
        if venue_text:
            reference.venue = Venue(name=venue_text, venue_type="unknown")
        
        if location.pages or location.volume:
            reference.location = location
    
    def _determine_reference_type(self, reference: Reference) -> ReferenceType:
        """Determine reference type based on parsed information."""
        if reference.venue and reference.venue.name:
            venue_name = reference.venue.name.lower()
            
            # Check for conference indicators
            conference_indicators = [
                "proceedings", "conference", "symposium", "workshop",
                "meeting", "congress", "summit", "forum", "acm", "ieee"
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
            
            # Check for book publishers
            book_indicators = [
                "springer", "elsevier", "wiley", "cambridge", "oxford",
                "academic press", "mit press", "university press"
            ]
            if any(indicator in venue_name for indicator in book_indicators):
                return ReferenceType.BOOK
        
        # Check for arXiv references
        if reference.title and "arxiv" in reference.title.lower():
            return ReferenceType.PREPRINT
        
        # Default to journal article
        return ReferenceType.JOURNAL_ARTICLE
    
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
        
        # Location (pages/volume): 10%
        if reference.location and (reference.location.pages or reference.location.volume):
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_field_confidences(self, reference: Reference) -> Dict[str, float]:
        """Calculate confidence scores for individual fields."""
        confidences = {}
        
        if reference.title:
            # High confidence for titles from structured HTML
            confidences["title"] = 0.95
        
        if reference.authors:
            # High confidence for authors from structured spans
            confidences["authors"] = 0.9
        
        if reference.venue_name:
            # Medium confidence for venue (parsed from unstructured text)
            confidences["venue"] = 0.7
        
        if reference.publication_year:
            # High confidence for years (regex pattern matching)
            confidences["year"] = 0.85
        
        if reference.location and reference.location.pages:
            # Medium confidence for pages
            confidences["pages"] = 0.75
        
        return confidences
    
    def _calculate_overall_confidence(self, references: List[Reference]) -> float:
        """Calculate overall confidence for the parsing result."""
        if not references:
            return 0.0
        
        individual_scores = [
            self._calculate_confidence(ref) for ref in references
        ]
        
        return sum(individual_scores) / len(individual_scores)