"""
PyMuPDF4LLM-based parser for fast PDF reference extraction.

This parser uses PyMuPDF4LLM to convert PDFs to markdown and then
extracts references using pattern matching. It provides significantly
faster processing compared to GROBID while maintaining reasonable accuracy.
"""

import re
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

try:
    import pymupdf4llm
except ImportError:
    pymupdf4llm = None

from .base_parser import BaseParser, ParseRequest, ParseResult, ParserCapabilities
from ..models.reference import Reference
from ..models.schemas import ParserType, ProcessingMetadata
from ..config.logging import get_logger


class PyMuPDFParser(BaseParser):
    """Fast PDF parser using PyMuPDF4LLM for markdown extraction."""
    
    def __init__(self, **config: Any):
        """Initialize PyMuPDF parser with configuration."""
        super().__init__(**config)
        
        if pymupdf4llm is None:
            raise ImportError(
                "pymupdf4llm is required for PyMuPDFParser. "
                "Install with: pip install pymupdf4llm"
            )
        
        # Configuration options
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.max_references = config.get("max_references", 200)
        self.include_raw_markdown = config.get("include_raw_markdown", False)
    
    @property
    def parser_type(self) -> ParserType:
        """Return parser type."""
        return ParserType.PYMUPDF
    
    @property  
    def capabilities(self) -> ParserCapabilities:
        """Return parser capabilities."""
        return ParserCapabilities(
            parser_type=ParserType.PYMUPDF,
            supported_content_types=["pdf"],
            supports_async=True,
            supports_batch=True,
            confidence_scoring=True,
            field_level_extraction=True,
            requires_external_service=False,
            service_dependencies=[],
            max_file_size_mb=50,
            max_processing_time_seconds=30,
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
        """Initialize parser - PyMuPDF4LLM requires no setup."""
        self.logger.info("PyMuPDF4LLM parser initialized")
    
    async def cleanup(self) -> None:
        """Cleanup - no resources to clean."""
        self.logger.info("PyMuPDF4LLM parser cleanup completed")
    
    async def _parse_implementation(self, request: ParseRequest) -> ParseResult:
        """Extract references from PDF using PyMuPDF4LLM."""
        
        start_time = time.time()
        
        # Handle different content types
        pdf_path = await self._prepare_pdf_file(request.content)
        
        try:
            # Extract markdown with optimized settings for speed
            markdown_text = pymupdf4llm.to_markdown(
                str(pdf_path),
                page_chunks=False,  # Keep as single document
                write_images=False,  # Skip images for speed
                image_path=None,    # No image extraction
                dpi=150,           # Lower DPI for speed
            )
            
            # Extract references from markdown
            references = self._extract_references_from_markdown(
                markdown_text, 
                request.paper_id or "unknown"
            )
            
            # Calculate confidence based on extracted data
            confidence = self._calculate_confidence(markdown_text, references)
            
            # Create processing metadata
            processing_time = (time.time() - start_time) * 1000
            metadata = ProcessingMetadata(
                parser_type=self.parser_type,
                processing_time_ms=processing_time,
                confidence_score=confidence
            )
            
            # Include raw markdown if requested in parser options
            if self.include_raw_markdown and request.parser_options.get("include_raw_output", False):
                metadata.raw_output = markdown_text[:2000]  # Limit size
            
            self.logger.info(
                f"Successfully extracted {len(references)} references",
                extra={
                    "paper_id": request.paper_id,
                    "processing_time_ms": processing_time,
                    "confidence_score": confidence,
                    "reference_count": len(references)
                }
            )
            
            return ParseResult(
                references=references,
                processing_metadata=metadata
            )
            
        except Exception as e:
            raise RuntimeError(f"PyMuPDF4LLM parsing failed: {e}")
        
        finally:
            # Clean up temporary file if created
            if hasattr(self, '_temp_file') and self._temp_file:
                try:
                    Path(self._temp_file).unlink(missing_ok=True)
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temp file: {e}")
    
    async def _prepare_pdf_file(self, content: Union[str, bytes, Path]) -> Path:
        """Prepare PDF file for processing."""
        if isinstance(content, Path):
            return content
        
        elif isinstance(content, str):
            # Assume it's a file path
            return Path(content)
        
        elif isinstance(content, bytes):
            # Create temporary file for bytes content
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
                f.write(content)
                self._temp_file = f.name
                return Path(f.name)
        
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")
    
    def _extract_references_from_markdown(self, markdown: str, paper_id: str) -> List[Reference]:
        """Extract references from PyMuPDF4LLM markdown output."""
        
        # Find references section
        ref_section = self._find_references_section(markdown)
        if not ref_section:
            self.logger.warning("No references section found in markdown")
            return []
        
        # Split into individual references
        ref_lines = self._split_references(ref_section)
        self.logger.debug(f"Found {len(ref_lines)} potential references")
        
        references = []
        for i, ref_line in enumerate(ref_lines):
            if len(references) >= self.max_references:
                self.logger.warning(f"Reached maximum reference limit ({self.max_references})")
                break
                
            try:
                ref = self._parse_single_reference(ref_line, i + 1, paper_id)
                if ref and ref.title:  # Only include if we found a title
                    references.append(ref)
            except Exception as e:
                self.logger.debug(f"Failed to parse reference {i+1}: {e}")
        
        return references
    
    def _find_references_section(self, markdown: str) -> Optional[str]:
        """Find the references section in markdown."""
        
        # Common reference section headers (case insensitive)
        ref_patterns = [
            r'^# References\s*$',
            r'^## References\s*$', 
            r'^# Bibliography\s*$',
            r'^## Bibliography\s*$',
            r'^# REFERENCES\s*$',
            r'^## REFERENCES\s*$',
            r'^References\s*$',
            r'^REFERENCES\s*$'
        ]
        
        # Try each pattern
        for pattern in ref_patterns:
            match = re.search(pattern, markdown, re.IGNORECASE | re.MULTILINE)
            if match:
                # Extract everything after the header
                start_pos = match.end()
                
                # Find next major section header (optional)
                next_section = re.search(r'\n#+\s+[A-Z]', markdown[start_pos:])
                if next_section:
                    end_pos = start_pos + next_section.start()
                    return markdown[start_pos:end_pos].strip()
                else:
                    return markdown[start_pos:].strip()
        
        # Fallback: look for numbered references
        ref_match = re.search(r'\n\s*\[?1[\]\.\)]\s+', markdown)
        if ref_match:
            return markdown[ref_match.start():].strip()
        
        return None
    
    def _split_references(self, ref_section: str) -> List[str]:
        """Split references section into individual reference strings."""
        
        # Remove excessive whitespace
        ref_section = re.sub(r'\n\s*\n', '\n', ref_section)
        
        # Try different splitting patterns
        patterns = [
            r'\n\s*\[(\d+)\]',  # [1], [2], etc.
            r'\n\s*(\d+)\.',    # 1., 2., etc.
            r'\n\s*(\d+)\)',    # 1), 2), etc.
            r'\n\s*\[(\d+)\)',  # [1), [2), etc.
        ]
        
        for pattern in patterns:
            splits = re.split(pattern, ref_section)
            if len(splits) > 3:  # Found reasonable splits
                # Recombine with reference numbers
                references = []
                for i in range(1, len(splits), 2):
                    if i + 1 < len(splits):
                        ref_num = splits[i]
                        ref_text = splits[i + 1].strip()
                        if ref_text and len(ref_text) > 10:  # Minimum length check
                            references.append(f"[{ref_num}] {ref_text}")
                return references
        
        # Fallback: split by double newlines
        refs = ref_section.split('\n\n')
        return [ref.strip() for ref in refs if ref.strip() and len(ref.strip()) > 10]
    
    def _parse_single_reference(self, ref_text: str, ref_id: int, paper_id: str) -> Optional[Reference]:
        """Parse a single reference from text."""
        
        # Clean up the reference text
        ref_text = re.sub(r'\s+', ' ', ref_text.strip())
        
        # Extract common patterns
        title = self._extract_title(ref_text)
        authors = self._extract_authors(ref_text)
        year = self._extract_year(ref_text)
        arxiv_id = self._extract_arxiv_id(ref_text)
        doi = self._extract_doi(ref_text)
        venue = self._extract_venue(ref_text)
        
        # Skip if no meaningful content found
        if not title and not authors and not arxiv_id and not doi:
            return None
        
        # Calculate confidence based on extracted fields
        confidence = self._calculate_reference_confidence(
            title, authors, year, arxiv_id, doi
        )
        
        return Reference(
            id=f"{paper_id}_ref_{ref_id}",
            title=title,
            authors=authors,
            year=year,
            arxiv_id=arxiv_id,
            doi=doi,
            venue=venue,
            raw_text=ref_text,
            confidence_score=confidence
        )
    
    def _extract_title(self, ref_text: str) -> Optional[str]:
        """Extract title from reference text."""
        # Look for quoted titles
        quoted_title = re.search(r'["\'"](.*?)["\'"]\s*[,\.]', ref_text)
        if quoted_title:
            return quoted_title.group(1).strip()
        
        # Look for titles after author names (common pattern)
        # This is a simplified heuristic
        parts = ref_text.split('.')
        if len(parts) >= 2:
            # Often the second part is the title
            potential_title = parts[1].strip()
            if 10 <= len(potential_title) <= 200:  # Reasonable title length
                return potential_title
        
        return None
    
    def _extract_authors(self, ref_text: str) -> List[str]:
        """Extract authors from reference text."""
        # This is a simplified extraction - could be improved with NLP
        # Look for patterns like "A. Smith, B. Jones,"
        
        # Extract the first part before a period (often authors)
        first_part = ref_text.split('.')[0].strip()
        
        # Remove reference number if present
        first_part = re.sub(r'^\[\d+\]\s*', '', first_part)
        first_part = re.sub(r'^\d+[\.\)]\s*', '', first_part)
        
        # Split by 'and' or commas
        if ' and ' in first_part:
            authors = [a.strip() for a in first_part.split(' and ')]
        else:
            authors = [a.strip() for a in first_part.split(',')]
        
        # Filter reasonable author names (basic validation)
        valid_authors = []
        for author in authors:
            if 2 <= len(author) <= 50 and re.search(r'[A-Za-z]', author):
                valid_authors.append(author)
        
        return valid_authors[:10]  # Limit number of authors
    
    def _extract_year(self, ref_text: str) -> Optional[int]:
        """Extract publication year from reference text."""
        # Look for 4-digit years
        year_matches = re.findall(r'\b(19|20)\d{2}\b', ref_text)
        if year_matches:
            year = int(year_matches[0])
            # Validate reasonable year range
            if 1900 <= year <= 2030:
                return year
        return None
    
    def _extract_arxiv_id(self, ref_text: str) -> Optional[str]:
        """Extract arXiv ID from reference text."""
        # Match various arXiv ID formats
        patterns = [
            r'arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)',
            r'arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)',
            r'(\d{4}\.\d{4,5}(?:v\d+)?)'  # Standalone format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, ref_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_doi(self, ref_text: str) -> Optional[str]:
        """Extract DOI from reference text."""
        # Match DOI patterns
        doi_match = re.search(r'doi:?\s*(10\.\d+/[^\s,\]]+)', ref_text, re.IGNORECASE)
        if doi_match:
            return doi_match.group(1)
        
        # Match DOI URLs
        doi_url_match = re.search(r'https?://(?:dx\.)?doi\.org/(10\.\d+/[^\s,\]]+)', ref_text, re.IGNORECASE)
        if doi_url_match:
            return doi_url_match.group(1)
        
        return None
    
    def _extract_venue(self, ref_text: str) -> Optional[str]:
        """Extract publication venue from reference text."""
        # This is very simplified - real venue extraction is complex
        # Look for text after title that might be venue
        
        # Common venue indicators
        venue_patterns = [
            r'In\s+([^,\.]+(?:Conference|Workshop|Symposium|Journal)[^,\.]*)',
            r'([A-Z][^,\.]*(?:Conference|Workshop|Symposium|Journal)[^,\.]*)',
            r'In\s+Proceedings\s+of\s+([^,\.]+)',
        ]
        
        for pattern in venue_patterns:
            match = re.search(pattern, ref_text)
            if match:
                venue = match.group(1).strip()
                if 5 <= len(venue) <= 100:  # Reasonable venue length
                    return venue
        
        return None
    
    def _calculate_reference_confidence(
        self, 
        title: Optional[str], 
        authors: List[str], 
        year: Optional[int],
        arxiv_id: Optional[str], 
        doi: Optional[str]
    ) -> float:
        """Calculate confidence score for a reference."""
        score = 0.0
        
        # Title contributes most to confidence
        if title and len(title) > 10:
            score += 0.4
        
        # Authors contribute significantly
        if authors:
            score += min(0.3, 0.1 * len(authors))
        
        # Year is important
        if year:
            score += 0.2
        
        # Strong identifiers boost confidence
        if arxiv_id:
            score += 0.3
        if doi:
            score += 0.3
        
        return min(1.0, score)  # Cap at 1.0
    
    def _calculate_confidence(self, markdown_text: str, references: List[Reference]) -> float:
        """Calculate overall parsing confidence."""
        if not references:
            return 0.0
        
        # Base confidence on number of references found
        ref_count_score = min(1.0, len(references) / 20.0)  # 20+ refs = full score
        
        # Average reference confidence
        ref_confidences = [ref.confidence_score for ref in references if ref.confidence_score]
        avg_ref_confidence = sum(ref_confidences) / len(ref_confidences) if ref_confidences else 0.0
        
        # Markdown quality indicators
        markdown_quality = 0.5  # Base score
        if "References" in markdown_text or "REFERENCES" in markdown_text:
            markdown_quality += 0.2
        if len(markdown_text) > 1000:  # Reasonable document length
            markdown_quality += 0.2
        if re.search(r'\[\d+\]', markdown_text):  # Numbered references
            markdown_quality += 0.1
        
        # Combine scores
        final_confidence = (ref_count_score * 0.3 + avg_ref_confidence * 0.5 + markdown_quality * 0.2)
        return min(1.0, final_confidence)